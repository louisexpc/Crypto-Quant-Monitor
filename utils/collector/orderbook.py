# bitmex_l2_to_hourly_features.py
import pandas as pd
from pathlib import Path
import numpy as np

def load_bitmex_l2_1day(path_gz: str, top_n: int = 10) -> pd.DataFrame:
    # BitMEX orderBookL2_25: columns 常見 ['symbol','id','side','size','price','timestamp']
    df = pd.read_csv(path_gz, compression="gzip")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"]  = pd.to_numeric(df["size"], errors="coerce")
    df = df.dropna(subset=["price","size"])

    # 每秒快照 → 取該秒的 top-N
    def one_second_feats(g):
        # g 為該秒的多行(雙邊多層)
        bids = g[g["side"].str.upper()=="BUY"].sort_values("price", ascending=False).head(top_n)
        asks = g[g["side"].str.upper()=="SELL"].sort_values("price", ascending=True ).head(top_n)
        if bids.empty or asks.empty:
            return pd.Series({"mid": np.nan, "spread_bps": np.nan, "imb": np.nan,
                              "bid_depth": np.nan, "ask_depth": np.nan})
        best_bid = bids["price"].iloc[0]; best_ask = asks["price"].iloc[0]
        mid = 0.5*(best_bid + best_ask)
        spread_bps = (best_ask - best_bid) / mid * 1e4
        bid_depth = bids["size"].sum()
        ask_depth = asks["size"].sum()
        imb = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth+ask_depth)>0 else np.nan
        return pd.Series({"mid": mid, "spread_bps": spread_bps, "imb": imb,
                          "bid_depth": bid_depth, "ask_depth": ask_depth})

    sec = df.groupby(pd.Grouper(key="timestamp", freq="1s", origin="epoch", label="right")).apply(one_second_feats)
    sec = sec.reset_index().rename(columns={"timestamp":"ts"})
    return sec

def minute_to_hourly(min_df: pd.DataFrame) -> pd.DataFrame:
    # 先把秒聚合到分（這裡對秒級特徵再平均即可）
    min_df = min_df.set_index("ts").resample("1min", label="right").mean()
    # 再聚合到 1H：平均 & 極值
    h = min_df.resample("1h", label="right").agg({
        "mid":        "mean",
        "spread_bps": ["mean","max"],
        "imb":        ["mean","median"],
        "bid_depth":  "mean",
        "ask_depth":  "mean",
    })
    h.columns = ["_".join(c for c in col if c) for col in h.columns.to_flat_index()]
    h = h.reset_index().rename(columns={"ts":"datetime_utc"})
    # 生成 Binance 用的 join key（UTC ms 的 bar close 時間）
    h["timestamp"] = (h["datetime_utc"].view("int64") // 1_000_000).astype("int64")
    return h

# 讀多天 → 串起來
def build_bitmex_hourly_features(gz_paths: list[str], top_n=10) -> pd.DataFrame:
    frames = []
    for p in sorted(gz_paths):
        sec = load_bitmex_l2_1day(p, top_n=top_n)
        h   = minute_to_hourly(sec)
        frames.append(h)
    out = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    # （可選）做 rolling 標準化減少跨日分佈漂移
    for c in ["spread_bps_mean","imb_mean","bid_depth_mean","ask_depth_mean"]:
        if c in out:
            m = out[c].rolling(24, min_periods=8).mean()
            s = out[c].rolling(24, min_periods=8).std()
            out[c+"_z24"] = (out[c]-m)/(s+1e-9)
    return out[["timestamp","spread_bps_mean","spread_bps_max","imb_mean","imb_median",
                "bid_depth_mean","ask_depth_mean","spread_bps_mean_z24","imb_mean_z24"]]

import pandas as pd

binance_1h = pd.read_csv("data/ohlcv_2023/binanceusdm_BTCUSDT_1h_with_m15_funding_ts.csv")
# 確保 join key：UTC ms 的收盤時間
binance_1h["timestamp"] = binance_1h["timestamp"].astype("int64")

bitmex_h = build_bitmex_hourly_features(sorted(list_of_gz_paths))
merged = pd.merge(binance_1h, bitmex_h, on="timestamp", how="left", validate="one_to_one")
merged.to_parquet("data/merged/BINANCE_with_BITMEX_LOB_1h.parquet", index=False)
print(merged.shape)