# trades_to_1min_features.py
# 需求：pandas >= 1.5, numpy
# 用法：python trades_to_1min_features.py
# 會產出：data/derived/btcusdt_trades_1min_features.csv

import os
import re
import gzip
import glob
import math
import numpy as np
import pandas as pd
from io import BytesIO
from zipfile import ZipFile

# ===== 可調參數 =====
INPUT_DIR = "data/binance_trades/BTCUSDT"
FILE_PATTERN = os.path.join(INPUT_DIR, "BTCUSDT-trades-*.zip")
START_DATE = "2023-01-01"
END_DATE   = "2025-08-23"  # 含當日
OUTPUT_CSV = "data/derived/btcusdt_trades_1min_features.csv"

# 高階特徵參數
VOL_SPIKE_WIN = 20          # volume_spike 參考的rolling視窗（分鐘）
TREND_SLOPE_WIN = 5         # trend_slope 參考最近N分鐘的線性迴歸斜率
PRICE_JUMP_THRESH = 0.001   # 價格跳動門檻（以 |log return| > 0.1% 為例）

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

DATE_RE = re.compile(r".*trades-(\d{4}-\d{2}-\d{2})\.zip$", re.IGNORECASE)

def _read_zip_as_df(zip_path: str) -> pd.DataFrame:
    """讀取單一 zip 檔內的 CSV / CSV.GZ 為 DataFrame。"""
    with ZipFile(zip_path) as zf:
        # 找第一個 .csv 或 .csv.gz 檔
        names = zf.namelist()
        csv_name = None
        for n in names:
            nl = n.lower()
            if nl.endswith(".csv") or nl.endswith(".csv.gz"):
                csv_name = n
                break
        if csv_name is None:
            raise FileNotFoundError(f"No CSV/CSV.GZ found in {zip_path}")

        with zf.open(csv_name) as f:
            if csv_name.lower().endswith(".gz"):
                with gzip.GzipFile(fileobj=f) as gz:
                    df = pd.read_csv(gz)
            else:
                df = pd.read_csv(f)
    return df

def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """欄位型別與清理。"""
    # time 可能是科學記號字串，先轉 float 再轉 int
    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("float64")
    df = df[df["time"].notna()]
    df["time"] = df["time"].astype("int64")

    # 其他數值
    for c in ["price", "qty", "quote_qty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # is_buyer_maker 轉 bool（TRUE/FALSE/true/false/1/0 都兜住）
    if df["is_buyer_maker"].dtype != bool:
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    df["is_buyer_maker"] = df["is_buyer_maker"].fillna(False)

    # 轉 UTC 分鐘
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["minute"] = df["ts"].dt.floor("min")
    return df

def _aggregate_minute(df: pd.DataFrame) -> pd.DataFrame:
    """把逐筆 trades 聚合成每分鐘特徵。"""
    # 準備加權項
    df["px_qty"] = df["price"] * df["qty"]

    # 買賣切分（Binance: is_buyer_maker=True 表示買方是掛單 -> 侵略方為賣方）
    # 為了符合一般「主動買」定義：buy = ~is_buyer_maker
    df["is_buy_taker"] = ~df["is_buyer_maker"]

    grp = df.groupby("minute", as_index=True)

    # 價格與成交量統計
    price_mean  = grp["price"].mean().rename("price_mean")
    price_max   = grp["price"].max().rename("price_max")
    price_min   = grp["price"].min().rename("price_min")
    price_std   = grp["price"].std(ddof=0).rename("price_std")
    price_range = (price_max - price_min).rename("price_range")

    vwap = (grp["px_qty"].sum() / grp["qty"].sum()).rename("price_vwap")

    # 量與頻率
    trade_count    = grp.size().rename("trade_count")
    qty_sum        = grp["qty"].sum().rename("qty_sum")
    quote_qty_sum  = grp["quote_qty"].sum().rename("quote_qty_sum")
    qty_mean       = grp["qty"].mean().rename("qty_mean")
    qty_max       = grp["qty"].max().rename("qty_max")
    qty_std       = grp["qty"].std(ddof=0).rename("qty_std")
    qty_skew      = grp["qty"].skew().rename("qty_skew")
    qty_kurt      = grp["qty"].apply(pd.Series.kurt).rename("qty_kurt")

    # 買賣方向特徵
    buy_qty   = grp.apply(lambda x: x.loc[x["is_buy_taker"], "qty"].sum()).rename("buy_qty")
    sell_qty  = grp.apply(lambda x: x.loc[~x["is_buy_taker"], "qty"].sum()).rename("sell_qty")
    buy_count = grp.apply(lambda x: x["is_buy_taker"].sum()).rename("buy_count")
    sell_count= (trade_count - buy_count).rename("sell_count")

    total_qty = qty_sum.replace(0, np.nan)
    buy_ratio       = (buy_qty / total_qty).rename("buy_ratio")
    buy_count_ratio = (buy_count / trade_count.replace(0, np.nan)).rename("buy_count_ratio")
    imbalance       = ((buy_qty - sell_qty) / total_qty).rename("imbalance")

    # 組回 DataFrame
    feat = pd.concat([
        price_mean, vwap, price_max, price_min, price_std, price_range,
        trade_count, qty_sum, quote_qty_sum, qty_mean, qty_max, qty_std, qty_skew, qty_kurt,
        buy_qty, sell_qty, buy_count, sell_count, buy_ratio, buy_count_ratio, imbalance
    ], axis=1)

    # 其他推導（需要先有 vwap / std 等）
    feat["volatility_proxy"] = feat["price_std"] / feat["price_vwap"]

    # 易讀的時間欄
    feat = feat.sort_index()
    feat["time_utc"] = feat.index
    return feat

def _compute_advanced_rolls(full_df: pd.DataFrame) -> pd.DataFrame:
    """在全區間上計算 volume_spike / price_jump_flag / trend_slope。"""
    df = full_df.copy()

    # volume_spike：與過去 N 分鐘平均量比較
    roll_mean_vol = df["qty_sum"].rolling(VOL_SPIKE_WIN, min_periods=1).mean()
    df["volume_spike"] = (df["qty_sum"] / roll_mean_vol) - 1.0

    # price_jump_flag：|log return| 是否超過門檻
    log_vwap = np.log(df["price_vwap"].replace(0, np.nan))
    abs_lr = log_vwap.diff().abs()
    df["price_jump_flag"] = (abs_lr > PRICE_JUMP_THRESH).astype(int)

    # trend_slope（對最近 N 分鐘 vwap 線性回歸的斜率）
    # 解析式：slope = cov(x,y)/var(x)，x = 0..N-1
    N = TREND_SLOPE_WIN
    x = np.arange(N)
    x_mean = (N - 1) / 2.0
    var_x = np.sum((x - x_mean)**2)

    def _slope(vals):
        # vals: array of vwap
        y = np.array(vals, dtype=float)
        y_mean = y.mean()
        cov = np.sum((x - x_mean) * (y - y_mean))
        return cov / var_x if var_x != 0 else np.nan

    df["trend_slope"] = (
        df["price_vwap"]
        .rolling(window=N, min_periods=N)
        .apply(_slope, raw=True)
    )

    return df

def _extract_date(zip_path: str):
    m = DATE_RE.match(zip_path)
    return m.group(1) if m else None

def main():
    files = sorted(glob.glob(FILE_PATTERN))
    # 篩日期區間
    chosen = []
    for p in files:
        d = _extract_date(p)
        if not d:
            continue
        if d >= START_DATE and d <= END_DATE:
            chosen.append(p)

    if not chosen:
        raise FileNotFoundError("找不到符合日期範圍的 zip 檔案，請確認路徑與檔名。")

    per_minute_parts = []
    for i, zp in enumerate(chosen, 1):
        print(f"[{i}/{len(chosen)}] Processing {os.path.basename(zp)} ...")
        try:
            raw = _read_zip_as_df(zp)
            raw = _coerce_schema(raw)
            part = _aggregate_minute(raw)
            per_minute_parts.append(part)
        except Exception as e:
            print(f"  -> 跳過（讀取/聚合失敗）：{zp} | err={e}")

    if not per_minute_parts:
        raise RuntimeError("沒有能成功聚合的分鐘資料。")

    # 合併所有日的分鐘資料
    full = pd.concat(per_minute_parts, axis=0)
    # 若有重複分鐘（理論上不會），保留最後一筆
    full = full[~full.index.duplicated(keep="last")]
    full = full.sort_index()

    # 整段計算進階rolling特徵
    full = _compute_advanced_rolls(full)

    # 欄位順序（依你列出）
    cols_order = [
        # (1) 價格與成交量統計
        "price_mean", "price_vwap", "price_max", "price_min", "price_std", "price_range",
        # (2) 成交量與頻率特徵
        "trade_count", "qty_sum", "quote_qty_sum", "qty_mean", "qty_max", "qty_std", "qty_skew", "qty_kurt",
        # (3) 買賣方向
        "buy_qty", "sell_qty", "buy_count", "sell_count", "buy_ratio", "buy_count_ratio", "imbalance",
        # (4) 高級統計與推導
        "volatility_proxy", "volume_spike", "price_jump_flag", "trend_slope",
        # 時間
        "time_utc"
    ]
    # 只輸出存在的欄位（避免極端情況缺少）
    cols_order = [c for c in cols_order if c in full.columns]
    out = full[cols_order].reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"完成！輸出：{OUTPUT_CSV}  共 {len(out):,} 列")

if __name__ == "__main__":
    main()
