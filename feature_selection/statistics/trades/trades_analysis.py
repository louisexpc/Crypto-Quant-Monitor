#!/usr/bin/env python3
# feature_selection/statistics/trades/trades_analysis.py
from __future__ import annotations

import argparse
import gzip
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd


DEFAULT_TRADES_DIR = Path("data/binance_trades/BTCUSDT")
DEFAULT_OUTPUT_DIR = Path("data/derived")

DATE_RE = re.compile(r".*trades-(\d{4}-\d{2}-\d{2})\.zip$", re.IGNORECASE)

# 這些欄位會填 0（沒有交易的 k-min bar）
ZERO_FILL_COLS = {
    "trade_count",
    "buy_volume",
    "sell_volume",
    "buy_trades",
    "sell_trades",
    "rv_var",
    "hl_range",
}
# 這些欄位保留 NaN（沒有交易的 k-min bar）
NAN_KEEP_COLS = {
    "vwap",
    "mean_intertrade_time",
    "max_intertrade_time",
}

# 這些欄位建議保持整數（fill missing 後會轉回 int）
COUNT_COLS = {"trade_count", "buy_trades", "sell_trades"}


def _find_trade_files(input_dir: Path, start: str | None, end: str | None) -> List[Path]:
    """
    1. 說明: 在資料夾中找出符合日期範圍的 Binance trades zip
    2. inputs:
       - input_dir: trades zip 目錄
       - start: YYYY-MM-DD（含）
       - end: YYYY-MM-DD（含）
    3. return:
       - files: List[Path]
    """
    files = sorted(input_dir.glob("*.zip"))
    chosen: List[Path] = []
    for fp in files:
        m = DATE_RE.match(fp.name)
        if not m:
            continue
        d = m.group(1)
        if start and d < start:
            continue
        if end and d > end:
            continue
        chosen.append(fp)
    return chosen


def _read_zip(zip_path: Path) -> pd.DataFrame:
    """
    1. 說明: 讀取單一 zip（內含 csv 或 csv.gz）並回傳 DataFrame
    2. inputs:
       - zip_path: zip 路徑
    3. return:
       - df: trades 原始資料
    """
    with ZipFile(zip_path) as zf:
        names = zf.namelist()
        csv_name = None
        for n in names:
            ln = n.lower()
            if ln.endswith(".csv") or ln.endswith(".csv.gz"):
                csv_name = n
                break
        if not csv_name:
            raise FileNotFoundError(f"No CSV/CSV.GZ found in {zip_path}")

        with zf.open(csv_name) as f:
            if csv_name.lower().endswith(".gz"):
                with gzip.GzipFile(fileobj=f) as gz:
                    df = pd.read_csv(gz)
            else:
                df = pd.read_csv(f)
    return df


def _clean_trades(df: pd.DataFrame, k_min: int) -> pd.DataFrame:
    """
    1. 說明: 清理 trades 原始表（型別、無效值、時間戳、衍生欄位），並產生 k-min bar_time
    2. inputs:
       - df: 原始 trades
       - k_min: 壓縮的分鐘數（k >= 1）
    3. return:
       - df: 清理後 trades（含 bar_time / is_buy / px_qty / ts）
    """
    if int(k_min) < 1:
        raise ValueError(f"k_min must be >= 1, got {k_min}")

    df = df.copy()

    # time sometimes comes in scientific notation
    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("float64")
    df = df[df["time"].notna()]
    df["time"] = df["time"].astype("int64")

    for col in ["price", "qty", "quote_qty"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["price"] > 0) & (df["qty"] > 0)]

    if "is_buyer_maker" in df.columns and df["is_buyer_maker"].dtype != bool:
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    df["is_buyer_maker"] = df.get("is_buyer_maker", False)
    df["is_buyer_maker"] = df["is_buyer_maker"].fillna(False).astype(bool)

    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.sort_values("ts")

    # k-min bucket
    df["bar_time"] = df["ts"].dt.floor(f"{int(k_min)}min")

    # Binance: is_buyer_maker=True => buyer is maker => aggressive side is SELL
    df["is_buy"] = ~df["is_buyer_maker"]
    df["px_qty"] = df["price"] * df["qty"]
    return df


def _realized_var(prices: pd.Series) -> float:
    """
    1. 說明: 以交易序列價格計算 realized variance（sum of squared log returns）
    2. inputs:
       - prices: 同一個 bar 內的 trade price 序列
    3. return:
       - rv: float
    """
    arr = prices.to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    rets = np.diff(np.log(arr))
    return float(np.sum(rets * rets))


def _mean_intertrade(ts: pd.Series) -> float:
    """
    1. 說明: 同一個 bar 內 trades 的平均間隔秒數
    2. inputs:
       - ts: 同一個 bar 內 trade timestamp（datetime64）
    3. return:
       - mean_sec: float
    """
    if len(ts) <= 1:
        return np.nan
    arr = ts.astype("int64")
    diffs_sec = np.diff(arr) / 1e9
    return float(np.mean(diffs_sec)) if len(diffs_sec) else np.nan


def _max_intertrade(ts: pd.Series) -> float:
    """
    1. 說明: 同一個 bar 內 trades 的最大間隔秒數
    2. inputs:
       - ts: 同一個 bar 內 trade timestamp（datetime64）
    3. return:
       - max_sec: float
    """
    if len(ts) <= 1:
        return np.nan
    arr = ts.astype("int64")
    diffs_sec = np.diff(arr) / 1e9
    return float(np.max(diffs_sec)) if len(diffs_sec) else np.nan


def _aggregate_per_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. 說明: 將 cleaned trades 聚合成每個 k-min bar 的特徵（不輸出 OHLCV 欄位）
    2. inputs:
       - df: cleaned trades（含 bar_time/is_buy/px_qty/ts）
    3. return:
       - feat: index=bar_time 的特徵表（欄位不帶 _1m/_5m 後綴）
    """
    grp = df.groupby("bar_time", sort=True)

    trade_cnt = grp.size().rename("trade_count")

    # vwap = sum(px*qty)/sum(qty)
    sum_qty = grp["qty"].sum()
    vwap = (grp["px_qty"].sum() / sum_qty.replace(0, np.nan)).rename("vwap")

    # buy/sell volume
    buy_vol = df.loc[df["is_buy"]].groupby("bar_time")["qty"].sum().rename("buy_volume")
    sell_vol = df.loc[~df["is_buy"]].groupby("bar_time")["qty"].sum().rename("sell_volume")

    buy_trades = grp["is_buy"].sum().rename("buy_trades")
    sell_trades = (trade_cnt - buy_trades).rename("sell_trades")

    # align missing bars (no buy / no sell)
    buy_vol = buy_vol.reindex(trade_cnt.index, fill_value=0.0)
    sell_vol = sell_vol.reindex(trade_cnt.index, fill_value=0.0)
    buy_trades = buy_trades.reindex(trade_cnt.index, fill_value=0)
    sell_trades = sell_trades.reindex(trade_cnt.index, fill_value=0)

    rv_var = grp["price"].apply(_realized_var).rename("rv_var")

    # hl_range: max(price) - min(price)
    hl_range = (grp["price"].max() - grp["price"].min()).rename("hl_range")

    mean_intertrade = grp["ts"].apply(_mean_intertrade).rename("mean_intertrade_time")
    max_intertrade = grp["ts"].apply(_max_intertrade).rename("max_intertrade_time")

    feat = pd.concat(
        [
            vwap,
            trade_cnt,
            buy_vol,
            sell_vol,
            buy_trades,
            sell_trades,
            rv_var,
            hl_range,
            mean_intertrade,
            max_intertrade,
        ],
        axis=1,
    ).sort_index()

    return feat


def _date_from_zipname(zip_path: Path) -> str:
    """
    1. 說明: 從 zip 檔名抽出 YYYY-MM-DD
    2. inputs:
       - zip_path: trades zip path
    3. return:
       - date_str: YYYY-MM-DD
    """
    m = DATE_RE.match(zip_path.name)
    if not m:
        raise ValueError(f"Zip name does not match expected pattern: {zip_path.name}")
    return m.group(1)


def _process_one_zip(zip_path: Path, k_min: int) -> Tuple[str, pd.DataFrame]:
    """
    1. 說明: 單一 zip -> read/clean/aggregate(k-min)，回傳 (date_str, per_bar_df)
    2. inputs:
       - zip_path: trades zip
       - k_min: 壓縮的分鐘數
    3. return:
       - date_str: YYYY-MM-DD
       - df_bar: index=bar_time 的 per-bar features
    """
    date_str = _date_from_zipname(zip_path)
    raw = _read_zip(zip_path)
    cleaned = _clean_trades(raw, k_min=int(k_min))
    agg = _aggregate_per_bar(cleaned)
    return date_str, agg


def _ensure_full_time_index(
    df_bar: pd.DataFrame, k_min: int, start_utc: Optional[pd.Timestamp], end_utc: Optional[pd.Timestamp]
) -> pd.DataFrame:
    """
    1. 說明: 將 per-bar 特徵補齊連續 bar_time index（缺的 bar 補出來）
    2. inputs:
       - df_bar: index=bar_time
       - k_min: 壓縮的分鐘數
       - start_utc: 可選，強制起始時間（UTC）
       - end_utc: 可選，強制結束時間（UTC）
    3. return:
       - df_full: 補齊後的 df
    """
    if df_bar.empty:
        return df_bar

    if int(k_min) < 1:
        raise ValueError(f"k_min must be >= 1, got {k_min}")

    idx_min = df_bar.index.min()
    idx_max = df_bar.index.max()
    if start_utc is not None:
        idx_min = min(idx_min, start_utc)
    if end_utc is not None:
        idx_max = max(idx_max, end_utc)

    full_idx = pd.date_range(idx_min.floor("min"), idx_max.floor("min"), freq=f"{int(k_min)}min", tz="UTC")
    df_full = df_bar.reindex(full_idx)

    for c in ZERO_FILL_COLS:
        if c in df_full.columns:
            df_full[c] = df_full[c].fillna(0.0)

    # counts back to int if present
    for c in COUNT_COLS:
        if c in df_full.columns:
            df_full[c] = df_full[c].astype("int64")

    # NAN_KEEP_COLS: keep NaN
    return df_full


def _reset_index_as_date(df_bar: pd.DataFrame) -> pd.DataFrame:
    """
    1. 說明: 安全地把 df_bar 的 index reset 成欄位，並統一命名為 date
       - 不假設 reset_index() 的欄名是 index/bar_time/...，一律取第一欄改成 date
    2. inputs:
       - df_bar: index 為 datetime 的 per-bar features
    3. return:
       - out: 包含 date 欄位的 DataFrame
    """
    out = df_bar.reset_index()
    idx_col = out.columns[0]
    return out.rename(columns={idx_col: "date"})


def _write_day_feather(out_dir: Path, date_str: str, k_min: int, df_bar: pd.DataFrame) -> Path:
    """
    1. 說明: 將單日結果寫成 feather（供 resume/最後合併）
    2. inputs:
       - out_dir: 暫存資料夾
       - date_str: YYYY-MM-DD
       - k_min: 壓縮的分鐘數
       - df_bar: index=bar_time 的 per-bar features
    3. return:
       - out_path: feather path
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trades_{int(k_min)}m_{date_str}.feather"

    tmp = _reset_index_as_date(df_bar)
    tmp["date"] = pd.to_datetime(tmp["date"], utc=True)
    tmp.to_feather(out_path)
    return out_path


def _finalize_and_save(df_bar: pd.DataFrame, output_csv: Path) -> None:
    """
    1. 說明: 組出最終輸出（date + features），存 CSV + Feather
       - 不再輸出 timestamp 欄位（只保留 date）
    2. inputs:
       - df_bar: index=bar_time 的 per-bar features
       - output_csv: 輸出 CSV 路徑
    3. return:
       - None
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    out = _reset_index_as_date(df_bar)

    # 強制 UTC dtype + 指定 isoformat（"2023-01-01 00:00:00+00:00"）
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    out["date"] = out["date"].str.replace(r"(\+|\-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)

    cols_order = [
        "date",
        "vwap",
        "trade_count",
        "buy_volume",
        "sell_volume",
        "buy_trades",
        "sell_trades",
        "rv_var",
        "hl_range",
        "mean_intertrade_time",
        "max_intertrade_time",
    ]
    existing_cols = [c for c in cols_order if c in out.columns]
    remaining = [c for c in out.columns if c not in existing_cols]
    out = out[existing_cols + remaining]

    out.to_csv(output_csv, index=False)
    output_feather = output_csv.with_suffix(".feather")
    out.to_feather(output_feather)

    print(f"Saved {len(out):,} rows -> {output_csv}")
    print(f"Saved {len(out):,} rows -> {output_feather}")


def _read_day_feather(fp: Path) -> pd.DataFrame:
    """
    1. 說明: 讀取單日 feather，並相容舊版欄名（date/minute/bar_time）
    2. inputs:
       - fp: feather path
    3. return:
       - df: index=date(UTC) 的 per-bar features
    """
    df = pd.read_feather(fp)

    time_col = None
    for cand in ("date", "bar_time", "minute"):
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        raise KeyError(f"{fp.name} missing time column. columns={df.columns.tolist()}")

    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.rename(columns={time_col: "date"}).set_index("date").sort_index()
    return df


def parse_args() -> argparse.Namespace:
    """
    1. 說明: CLI 參數
    2. inputs: None
    3. return: argparse.Namespace
    """
    ap = argparse.ArgumentParser(description="Aggregate Binance trades zips into k-minute trade-only features.")
    ap.add_argument("--trades-dir", type=Path, default=DEFAULT_TRADES_DIR, help="Directory containing daily trade zip files.")
    ap.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD, inclusive).")
    ap.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD, inclusive).")
    ap.add_argument("--k-min", type=int, default=1, help="Aggregate to k-minute bars (k >= 1).")
    ap.add_argument("--output", type=Path, default=None, help="Output CSV path (also writes .feather).")
    ap.add_argument("--max-workers", type=int, default=8, help="ProcessPool workers.")
    ap.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Temp dir for per-day feather. Default: <output_dir>/_tmp_trades_{k}m",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing per-day feather files in tmp-dir.",
    )
    ap.add_argument(
        "--fill-missing-bars",
        action="store_true",
        help="Reindex to continuous k-minute bars and fill missing bars (recommended for downstream ML).",
    )
    return ap.parse_args()


def main() -> None:
    """
    1. 說明: 主程式（平行化處理每日 zip -> 合併 -> 輸出 CSV+Feather）
    2. inputs: None
    3. return: None
    """
    args = parse_args()
    k_min = int(args.k_min)
    if k_min < 1:
        raise ValueError(f"--k-min must be >= 1, got {k_min}")

    trade_files = _find_trade_files(args.trades_dir, args.start, args.end)
    if not trade_files:
        raise FileNotFoundError("No trade zip files matched the given range.")

    output_csv = args.output
    if output_csv is None:
        output_csv = DEFAULT_OUTPUT_DIR / f"btcusdt_trades_{k_min}m_stats.csv"

    tmp_dir = args.tmp_dir or (output_csv.parent / f"_tmp_trades_{k_min}m")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(trade_files)} trade files in {args.trades_dir}")
    print(f"k_min={k_min}, max_workers={args.max_workers}, tmp_dir={tmp_dir}, resume={args.resume}")

    cached: dict[str, Path] = {}
    if args.resume:
        pat = re.compile(rf"trades_{k_min}m_(\d{{4}}-\d{{2}}-\d{{2}})\.feather$")
        for fp in sorted(tmp_dir.glob(f"trades_{k_min}m_*.feather")):
            m = pat.match(fp.name)
            if m:
                cached[m.group(1)] = fp

    to_run: List[Path] = []
    for z in trade_files:
        d = _date_from_zipname(z)
        if args.resume and d in cached:
            continue
        to_run.append(z)

    if to_run:
        futures = []
        with ProcessPoolExecutor(max_workers=int(args.max_workers)) as ex:
            for z in to_run:
                futures.append(ex.submit(_process_one_zip, z, k_min))

            done = 0
            total = len(to_run)
            for fu in as_completed(futures):
                date_str, df_day = fu.result()
                _write_day_feather(tmp_dir, date_str, k_min, df_day)
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"[progress] {done}/{total} days done")

    day_files = sorted(tmp_dir.glob(f"trades_{k_min}m_*.feather"))
    if not day_files:
        raise RuntimeError(f"No per-day feather files found in tmp_dir: {tmp_dir}")

    per_day_dfs: List[pd.DataFrame] = []
    for fp in day_files:
        per_day_dfs.append(_read_day_feather(fp))

    df_bar = pd.concat(per_day_dfs, axis=0).sort_index()
    df_bar = df_bar[~df_bar.index.duplicated(keep="last")]

    if args.fill_missing_bars:
        start_utc = pd.Timestamp(f"{args.start} 00:00:00+00:00") if args.start else None
        end_utc = pd.Timestamp(f"{args.end} 23:59:00+00:00") if args.end else None
        df_bar = _ensure_full_time_index(df_bar, k_min=k_min, start_utc=start_utc, end_utc=end_utc)

    _finalize_and_save(df_bar, output_csv)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()

"""
python feature_selection/statistics/trades/trades_analysis.py \
  --trades-dir data/binance_trades/BTCUSDT \
  --start 2023-01-01 \
  --end 2025-12-31 \
  --k-min 1 \
  --output data/derived/btcusdt_trades_1m_stats_230101-251231.csv \
  --max-workers 16 \
  --fill-missing-bars \
  --resume
"""
