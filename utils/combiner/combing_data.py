# combing_data_cols16.py  — 1H + 4×15m + funding（穩定版：searchsorted 映射）
# columns:
# [datetime,timestamp,open,high,low,close,volume,
#  m15_0_close,m15_0_vol,m15_1_close,m15_1_vol,m15_2_close,m15_2_vol,m15_3_close,m15_3_vol,funding_rate]

import numpy as np
import pandas as pd
from pathlib import Path

# -------- 路徑（依你的檔名） --------
slow_ohlcv_path    = Path(r"data/ohlcv_2023/binanceusdm_swap_BTC-USDT-USDT_1h.csv")
fast_ohlcv_path    = Path(r"data/ohlcv_2023/binanceusdm_swap_BTC-USDT-USDT_15m.csv")
funding_rate_path  = Path(r"data/ohlcv_2023/binanceusdm_BTCUSDT.csv")
out_path           = Path(r"data/ohlcv_2023/binanceusdm_BTCUSDT_1h_with_m15_funding_cols16.csv")

# -------- Helpers --------
def _read_with_timestamp(path: Path, kind: str) -> pd.DataFrame:
    """回傳含 UTC-ms 'timestamp' 的 DataFrame；欄名全小寫+去空白，timestamp 轉成 int64。"""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("Asia/Taipei")
    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = set(df.columns)

    if "timestamp" in cols:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        df["timestamp"] = (ts // 1).astype("Int64")
    elif kind == "funding" and "fundingtime" in cols:
        ft = pd.to_numeric(df["fundingtime"], errors="coerce").astype("Int64")
        # 去抖：落到秒再回毫秒
        df["timestamp"] = ((ft // 1000) * 1000).astype("Int64")
    else:
        ts_utc_ns = df.index.tz_convert("UTC").view("int64")
        df["timestamp"] = (ts_utc_ns // 1_000_000).astype("Int64")

    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df

def _assert_grid(df: pd.DataFrame, step_ms: int, name: str):
    """確保時間在固定格點上；若有偏移則 snap 回格點。"""
    off = (df["timestamp"] % step_ms).unique()
    if len(off) > 1 or (len(off) == 1 and off[0] != 0):
        print(f"[warn] {name} timestamps not perfectly on {step_ms}ms grid; snapping.")
        df["timestamp"] = ((df["timestamp"] // step_ms) * step_ms).astype("int64")
        df.sort_values("timestamp", inplace=True)
        df.drop_duplicates("timestamp", keep="last", inplace=True)

def _select_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    alias = {"open":"open","high":"high","low":"low","close":"close",
             "volume":"volume","vol":"volume","qty":"volume"}
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = df["timestamp"]
    for k,v in alias.items():
        if k in df.columns and v not in out.columns:
            out[v] = pd.to_numeric(df[k], errors="coerce")
    need = ["open","high","low","close","volume"]
    miss = [c for c in need if c not in out.columns]
    if miss: raise ValueError(f"Missing OHLCV columns: {miss}")
    return out[["timestamp","open","high","low","close","volume"]]

def attach_15m_blocks_by_ts(df1h: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    """在 1H 每列附上 4 根 15m（t-45, t-30, t-15, t）的 close/volume，精確對齊 timestamp。"""
    out = df1h.copy()
    f15 = df15.set_index("timestamp").sort_index()
    for j, off in enumerate([45,30,15,0]):  # minutes
        key_ts = out["timestamp"].astype("int64") - off * 60_000
        picked = f15.reindex(key_ts.values)
        out[f"m15_{j}_close"] = pd.to_numeric(picked["close"], errors="coerce").to_numpy()
        out[f"m15_{j}_vol"]   = pd.to_numeric(picked["volume"], errors="coerce").to_numpy()
    return out

def broadcast_funding_by_searchsorted(df1h: pd.DataFrame, df_f: pd.DataFrame) -> pd.DataFrame:
    """
    以 numpy.searchsorted 做「上一筆 funding」映射（等價 backward asof，超穩定）。
    容錯欄：fundingrate / funding_rate / fundingRate。
    """
    out = df1h.copy()

    # 找 funding 欄位
    cand = [c for c in df_f.columns if c.strip().lower() in ("fundingrate","funding_rate")]
    if not cand:
        raise ValueError("Funding file has no 'fundingRate'/'funding_rate' column.")
    f = df_f.rename(columns={cand[0]:"funding_rate"})[["timestamp","funding_rate"]].copy()
    f["funding_rate"] = pd.to_numeric(f["funding_rate"], errors="coerce")

    # 轉 numpy 陣列（升序）
    f = f.dropna(subset=["timestamp"]).sort_values("timestamp")
    f_ts = f["timestamp"].to_numpy(dtype=np.int64)
    f_fr = f["funding_rate"].to_numpy(dtype=float)

    x_ts = out["timestamp"].to_numpy(dtype=np.int64)

    # 對每個 1H 時點 t，找 f_ts 中 <= t 的最後一個索引
    # idx = searchsorted(f_ts, t, side='right') - 1；若 <0 表示還沒有 funding
    idx = np.searchsorted(f_ts, x_ts, side="right") - 1
    valid = idx >= 0
    funding = np.full(x_ts.shape, np.nan, dtype=float)
    funding[valid] = f_fr[idx[valid]]

    out["funding_rate"] = funding
    print(f"[info] funding_rate filled rows: {int(np.isfinite(funding).sum())} / {len(out)}")
    if not np.isfinite(funding).any():
        print(f"[hint] first funding ts (UTC ms): {f_ts.min() if f_ts.size else 'N/A'}",
              f"-> local: {pd.to_datetime(f_ts.min(), unit='ms', utc=True).tz_convert('Asia/Taipei') if f_ts.size else 'N/A'}")
    return out

# -------- Main --------
def main():
    # 1) 讀檔 + timestamp
    df1h_raw = _read_with_timestamp(slow_ohlcv_path, kind="ohlcv")
    df15_raw = _read_with_timestamp(fast_ohlcv_path, kind="ohlcv")
    df_f_raw = _read_with_timestamp(funding_rate_path, kind="funding")

    # 2) 選出 OHLCV 並檢查格點
    df1h = _select_ohlcv(df1h_raw)
    df15 = _select_ohlcv(df15_raw)
    _assert_grid(df1h, 3_600_000, "1H")
    _assert_grid(df15,   900_000, "15m")

    # 3) 1H + 4×15m（精確對齊）
    combo = attach_15m_blocks_by_ts(df1h, df15)

    # 4) funding 廣播（searchsorted 映射）
    combo = broadcast_funding_by_searchsorted(combo, df_f_raw)

    # 5) 排序、加 datetime(+08:00)、只留 16 欄
    combo = combo.sort_values("timestamp").reset_index(drop=True)

    # datetime：固定 +08:00 字串
    dt_local = pd.to_datetime(combo["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Taipei")
    dt_str = dt_local.dt.strftime('%Y-%m-%d %H:%M:%S%z').str.replace(r'([+-]\d{2})(\d{2})$', r'\1:\2', regex=True)
    combo.insert(0, "datetime", dt_str)

    wanted = [
        "datetime","timestamp","open","high","low","close","volume",
        "m15_0_close","m15_0_vol","m15_1_close","m15_1_vol",
        "m15_2_close","m15_2_vol","m15_3_close","m15_3_vol","funding_rate"
    ]
    for c in wanted:
        if c not in combo.columns:
            combo[c] = pd.NA
    combo = combo[wanted]

    # 6) 輸出
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combo.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}  (rows={len(combo)})")

if __name__ == "__main__":
    main()
