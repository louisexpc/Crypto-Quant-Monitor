import pandas as pd
import numpy as np
from pathlib import Path

# ==== 路徑 ====
p_15m = Path("data/ohlcv_2023/binanceusdm_swap_BTC-USDT-USDT_15m.csv")
p_1m  = Path("data/derived/btcusdt_trades_1min_features.csv")
p_out = Path("data/derived/btcusdt_15m_with_flat_1m.csv")

# ==== 參數 ====
MINUTE_STEPS = 15           # 取過去 15 分鐘（t-14 ... t）
BAR_IS_END   = False        # 若 15m 檔的時間欄位是「bar 結束時間」就設 True；若是「開始時間」就 False
TZ_15M       = "Asia/Taipei"  # 你這支 15m 檔案的 datetime 顯示 +08:00
FILL_POLICY  = "zero"       # 缺值處理： "zero" / "ffill" / "drop"

# ==== 讀 15m OHLCV（統一到 UTC 的 bar 結束時間） ====
df15 = pd.read_csv(p_15m)

# 優先用 timestamp（ms），其次用 datetime(+08:00)
if "timestamp" in df15.columns:
    t_utc = pd.to_datetime(df15["timestamp"], unit="ms", utc=True)
else:
    # 若只有 datetime(+08)，轉成 UTC
    t_lcl = pd.to_datetime(df15["datetime"]).dt.tz_convert(TZ_15M) \
            if pd.to_datetime(df15["datetime"]).dt.tz is not None \
            else pd.to_datetime(df15["datetime"]).dt.tz_localize(TZ_15M)
    t_utc = t_lcl.dt.tz_convert("UTC")

# 若是 bar 開始時間 → 加 15 分鐘得到結束時間
if not BAR_IS_END:
    t_utc = t_utc + pd.Timedelta(minutes=15)

df15.index = t_utc
df15.index.name = "time_utc"
# 你也可以在這裡先算 15m 指標，再一起 concat；這裡先只保留原 OHLCV
base_cols = ["open","high","low","close","volume"]
base_cols = [c for c in base_cols if c in df15.columns]
df15 = df15[base_cols].sort_index()

# ==== 讀 1m 特徵（索引改成 UTC 的分鐘結束時間） ====
df1m = pd.read_csv(p_1m)

# 強制把 time_utc 解析成 tz-aware UTC
if "time_utc" in df1m.columns:
    t1 = pd.to_datetime(df1m["time_utc"], utc=True)
else:
    raise ValueError("1m 檔缺少 time_utc 欄位")

df1m.index = t1
df1m.index.name = "time_utc"

# 只保留數值特徵欄（排除 time_utc）
minute_cols = [c for c in df1m.columns if c != "time_utc"]
# 轉為數值；無法轉換的一律 NaN
for c in minute_cols:
    df1m[c] = pd.to_numeric(df1m[c], errors="coerce")

df1m = df1m[minute_cols].sort_index()

# 補足每分鐘連續索引（可選）
full_idx = pd.date_range(start=df1m.index.min(), end=df1m.index.max(), freq="1min", tz="UTC")
df1m = df1m.reindex(full_idx)

# 缺值處理策略
if FILL_POLICY == "ffill":
    df1m = df1m.fillna(method="ffill").fillna(0.0)
elif FILL_POLICY == "zero":
    df1m = df1m.fillna(0.0)
elif FILL_POLICY == "drop":
    # 之後在組樣本時，若某個 t 的 15xFm 區塊有 NaN 就跳過該樣本
    pass
else:
    raise ValueError("未知 FILL_POLICY")

# ==== 打平 1m 區塊並與 15m 合併 ====
rows = []
times = []
flat_colnames = []
# 構造展平後欄名
for k in range(MINUTE_STEPS):
    offset = -(MINUTE_STEPS-1-k)   # -14, -13, ..., 0
    flat_colnames += [f"m_{offset}_{c}" for c in minute_cols]

for t, base_row in df15.iterrows():
    # 取 t-14m..t 的 1m 區間
    idx = pd.date_range(end=t, periods=MINUTE_STEPS, freq="1min", tz="UTC")
    block = df1m.reindex(idx)

    # 對 drop 策略：若這個區塊有 NaN 就跳過
    if FILL_POLICY == "drop" and block.isna().any().any():
        continue
    # 其他策略：把殘存 NaN 轉 0
    block = block.fillna(0.0)

    flat = block.to_numpy().reshape(-1)  # 長度 = 15 * minute_feat
    feat = np.concatenate([base_row.to_numpy(dtype=float), flat], axis=0)
    rows.append(feat)
    times.append(t)

# 合併表
out_cols = base_cols + flat_colnames
df_out = pd.DataFrame(rows, index=pd.Index(times, name="time_utc"), columns=out_cols).sort_index()

# ====（可選）建立 1H 標籤：以 15m close 作為基準 ====
# 這裡給一個簡單範例：未來 1H 報酬；若你要 2/3 分類，這段自己改
if "close" in df15.columns:
    future_close = df15["close"].reindex(df_out.index).shift(-4)  # 4 根 15m = 1H 之後的 close
    ret_1h = (future_close - df15["close"].reindex(df_out.index)) / df15["close"].reindex(df_out.index)
    df_out["y_ret_1h"] = ret_1h
    # 例如 3 分類：<-thr: down, |ret|<=thr: flat, >thr: up
    thr = 0.001  # 0.1% 例子
    conds = [ret_1h < -thr, ret_1h.abs() <= thr, ret_1h > thr]
    df_out["y_class_3"] = np.select(conds, [0, 1, 2], default=np.nan).astype("float")

# 存檔
p_out.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(p_out)
print(f"Saved: {p_out}  shape={df_out.shape}")
print("Minute feature count =", len(minute_cols), "| flattened part =", MINUTE_STEPS*len(minute_cols))
