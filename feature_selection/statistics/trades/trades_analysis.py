#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path
from typing import Iterable, List
from zipfile import ZipFile

import numpy as np
import pandas as pd


DEFAULT_TRADES_DIR = Path("data/binance_trades/BTCUSDT")
DEFAULT_OHLCV_1M = Path("data/ohlcv_2023/binanceusdm_swap_BTC-USDT-USDT_1m.csv")
DEFAULT_OUTPUT = Path("data/derived/btcusdt_trades_1m_stats.csv")
DATE_RE = re.compile(r".*trades-(\d{4}-\d{2}-\d{2})\.zip$", re.IGNORECASE)


def _find_trade_files(input_dir: Path, start: str | None, end: str | None) -> List[Path]:
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


def _clean_trades(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # time sometimes comes in scientific notation
    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("float64")
    df = df[df["time"].notna()]
    df["time"] = df["time"].astype("int64")

    for col in ["price", "qty", "quote_qty"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["price"] > 0) & (df["qty"] > 0)]

    if df["is_buyer_maker"].dtype != bool:
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    df["is_buyer_maker"] = df["is_buyer_maker"].fillna(False)

    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.sort_values("ts")
    df["minute"] = df["ts"].dt.floor("min")
    df["is_buy"] = ~df["is_buyer_maker"]
    df["px_qty"] = df["price"] * df["qty"]
    return df


def _realized_var(prices: pd.Series) -> float:
    arr = prices.to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    rets = np.diff(np.log(arr))
    return float(np.sum(rets * rets))


def _mean_intertrade(ts: pd.Series) -> float:
    if len(ts) <= 1:
        return np.nan
    arr = ts.astype("int64")
    diffs_sec = np.diff(arr) / 1e9
    return float(np.mean(diffs_sec)) if len(diffs_sec) else np.nan


def _max_intertrade(ts: pd.Series) -> float:
    if len(ts) <= 1:
        return np.nan
    arr = ts.astype("int64")
    diffs_sec = np.diff(arr) / 1e9
    return float(np.max(diffs_sec)) if len(diffs_sec) else np.nan


def _aggregate_per_minute(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("minute", sort=True)

    open_p = grp["price"].first().rename("open_1m")
    high_p = grp["price"].max().rename("high_1m")
    low_p = grp["price"].min().rename("low_1m")
    close_p = grp["price"].last().rename("close_1m")

    volume = grp["qty"].sum().rename("volume_1m")
    trade_cnt = grp.size().rename("trade_count_1m")
    vwap = (grp["px_qty"].sum() / volume.replace(0, np.nan)).rename("vwap_1m")

    buy_vol = grp.apply(lambda x: x.loc[x["is_buy"], "qty"].sum()).rename("buy_volume_1m")
    sell_vol = grp.apply(lambda x: x.loc[~x["is_buy"], "qty"].sum()).rename("sell_volume_1m")
    buy_trades = grp["is_buy"].sum().rename("buy_trades_1m")
    sell_trades = (trade_cnt - buy_trades).rename("sell_trades_1m")

    rv_var = grp["price"].apply(_realized_var).rename("rv_var_1m")
    hl_range = (high_p - low_p).rename("hl_range_1m")
    mean_intertrade = grp["ts"].apply(_mean_intertrade).rename("mean_intertrade_time_1m")
    max_intertrade = grp["ts"].apply(_max_intertrade).rename("max_intertrade_time_1m")

    feat = pd.concat(
        [
            open_p,
            high_p,
            low_p,
            close_p,
            volume,
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
    )
    feat = feat.sort_index()
    feat["timestamp"] = (feat.index.astype("int64") // 10**6).astype("int64")
    return feat


def _load_ohlcv(ohlcv_path: Path) -> pd.DataFrame:
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {ohlcv_path}")
    ohlcv = pd.read_csv(ohlcv_path)
    required = {"datetime", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"Missing columns in OHLCV file: {', '.join(sorted(missing))}")

    ohlcv = ohlcv.drop_duplicates(subset=["timestamp"])
    ohlcv["timestamp"] = pd.to_numeric(ohlcv["timestamp"], errors="coerce").astype("int64")
    renamed = ohlcv.rename(
        columns={
            "open": "open_1m",
            "high": "high_1m",
            "low": "low_1m",
            "close": "close_1m",
            "volume": "volume_1m",
        }
    )
    keep_cols = [
        "datetime",
        "timestamp",
        "open_1m",
        "high_1m",
        "low_1m",
        "close_1m",
        "volume_1m",
    ]
    return renamed[keep_cols].set_index("timestamp")


def _merge(ohlcv: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    merged = ohlcv.join(agg.set_index("timestamp"), how="left", lsuffix="_ohlcv", rsuffix="_trade")

    # Prefer trade-derived price/volume if present, otherwise keep OHLCV values.
    for col in ["open_1m", "high_1m", "low_1m", "close_1m", "volume_1m"]:
        trade_col = f"{col}_trade"
        ohlcv_col = f"{col}_ohlcv"
        if trade_col in merged.columns and ohlcv_col in merged.columns:
            merged[col] = merged[trade_col].combine_first(merged[ohlcv_col])
            merged = merged.drop(columns=[trade_col, ohlcv_col])
        elif trade_col in merged.columns:
            merged[col] = merged[trade_col]
            merged = merged.drop(columns=[trade_col])
        elif ohlcv_col in merged.columns:
            merged[col] = merged[ohlcv_col]
            merged = merged.drop(columns=[ohlcv_col])

    additive_cols = [
        "volume_1m",
        "trade_count_1m",
        "buy_volume_1m",
        "sell_volume_1m",
        "buy_trades_1m",
        "sell_trades_1m",
        "rv_var_1m",
        "hl_range_1m",
    ]
    for c in additive_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)

    merged = merged.sort_index()
    merged = merged.reset_index().rename(columns={"index": "timestamp"})
    cols_order = [
        "datetime",
        "timestamp",
        "open_1m",
        "high_1m",
        "low_1m",
        "close_1m",
        "volume_1m",
        "vwap_1m",
        "trade_count_1m",
        "buy_volume_1m",
        "sell_volume_1m",
        "buy_trades_1m",
        "sell_trades_1m",
        "rv_var_1m",
        "hl_range_1m",
        "mean_intertrade_time_1m",
        "max_intertrade_time_1m",
    ]
    existing_cols = [c for c in cols_order if c in merged.columns]
    remaining = [c for c in merged.columns if c not in existing_cols]
    return merged[existing_cols + remaining]


def _process_files(files: Iterable[Path]) -> pd.DataFrame:
    per_day: List[pd.DataFrame] = []
    for i, fp in enumerate(sorted(files), 1):
        print(f"[{i}] {fp.name}")
        raw = _read_zip(fp)
        cleaned = _clean_trades(raw)
        agg = _aggregate_per_minute(cleaned)
        per_day.append(agg)
    if not per_day:
        raise RuntimeError("No trade data aggregated.")
    out = pd.concat(per_day, axis=0)
    out = out[~out.index.duplicated(keep="last")]
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate Binance trades zips into 1-minute features.")
    ap.add_argument("--trades-dir", type=Path, default=DEFAULT_TRADES_DIR, help="Directory containing daily trade zip files.")
    ap.add_argument("--ohlcv-1m", type=Path, default=DEFAULT_OHLCV_1M, help="1m OHLCV CSV path.")
    ap.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD, inclusive).")
    ap.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD, inclusive).")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    return ap.parse_args()


def main():
    args = parse_args()
    trade_files = _find_trade_files(args.trades_dir, args.start, args.end)
    if not trade_files:
        raise FileNotFoundError("No trade zip files matched the given range.")

    print(f"Found {len(trade_files)} trade files in {args.trades_dir}")
    agg = _process_files(trade_files)

    ohlcv = _load_ohlcv(args.ohlcv_1m)
    merged = _merge(ohlcv, agg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Saved {len(merged):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
