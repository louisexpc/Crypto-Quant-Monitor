# FNG（日頻）→ 15m（UTC）→ CSV，含 datetime(ISO) + timestamp(Unix秒)
# 需求：pip install pandas requests

import os
from pathlib import Path
import requests
import pandas as pd
import numpy as np

START_DATE = "2022-12-31"

def load_fng_15m_utc(start_date: str = START_DATE) -> pd.DataFrame:
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    j = requests.get(url, timeout=30).json()
    data = j["data"]
    df = pd.DataFrame(data)

    # 1) 時間與型別
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.loc[start_date:]                   # 只留 2023-01-01 之後
    df["value"] = df["value"].astype(float)    # FNG 原始值（0~100）

    # 2) 轉 15m（UTC），前向填補
    fng_15m = df[["value"]].resample("15T").ffill().rename(columns={"value": "fng"})

    # 3) 衍生特徵（注意對日頻→15m後，多數時間 diff=0 屬正常現象）
    fng_15m["fng_diff1"] = fng_15m["fng"].diff()  # 15m 階段差
    roll = 24*7*4  # 7天×24h×每小時4個15m點
    fng_15m["fng_z7d"] = (
        fng_15m["fng"] - fng_15m["fng"].rolling(roll, min_periods=24).mean()
    ) / (fng_15m["fng"].rolling(roll, min_periods=24).std() + 1e-6)

    # 4) 防洩漏：整體往前移一格 15m
    # fng_15m = fng_15m.shift(1)

    # 5) 產生你要的欄位：datetime(ISO字串, UTC) + timestamp(Unix秒, int)
    out = fng_15m.copy()
    out = out.loc[out.index >= pd.to_datetime(start_date, utc=True)]
    out["datetime"]  = out.index.astype("datetime64[ns]")
    out["timestamp"] = (out.index.view("int64") // 10**9).astype("int64")  # ns→s
    # 置頂欄位順序
    out = out[["datetime", "timestamp", "fng", "fng_diff1", "fng_z7d"]]

    return out

# --- 執行並輸出 CSV ---
df = load_fng_15m_utc()
out_dir = Path("data/FNG"); out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / "fng_15m_utc.csv"
df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path.resolve()}")
print(df.tail(3))
