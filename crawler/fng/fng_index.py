#!/usr/bin/env python3
# crawler/fng/fng_index.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

DEFAULT_START = "2023-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_OUTDIR = Path("data/FNG")
DEFAULT_FILENAME = "fng_1d_utc.csv"


def load_fng_1d_utc(start_date: str = DEFAULT_START, end_date: str | None = None) -> pd.DataFrame:
    """
    1. 說明: 從 alternative.me 抓 Fear & Greed Index，並轉成 1D（UTC）時間序列
    2. inputs:
       - start_date: 起始日期（YYYY-MM-DD）
       - end_date: 結束日期（YYYY-MM-DD or None）
    3. return:
       - out: columns = ["date", "timestamp", "fng"]
              date 格式: "YYYY-MM-DD 00:00:00+00:00"（UTC）
              timestamp: Unix seconds
    """
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    j = requests.get(url, timeout=30).json()
    df = pd.DataFrame(j["data"])

    # API 的 timestamp 是 Unix seconds
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["fng"] = pd.to_numeric(df["value"], errors="coerce").astype(float)

    df = df.set_index("timestamp").sort_index()
    df = df.loc[start_date:]
    if end_date:
        df = df.loc[:end_date]

    # 轉成「每日 00:00:00+00:00」的 index（用 date 當天，對齊到日線）
    daily = df[["fng"]].resample("1D").ffill()

    out = daily.copy()
    out = out.loc[out.index >= pd.to_datetime(start_date, utc=True)]
    out = out.reset_index().rename(columns={"timestamp": "date"})

    # date 欄位格式固定：YYYY-MM-DD 00:00:00+00:00
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    out["date"] = out["date"].str.replace(r"(\+0000)$", r"+00:00", regex=True)

    # timestamp（Unix seconds）
    dt = pd.to_datetime(out["date"], utc=True)
    out["timestamp"] = (dt.astype("int64") // 10**9).astype("int64")

    out = out[["date", "timestamp", "fng"]]
    return out


def parse_args() -> argparse.Namespace:
    """
    1. 說明: CLI 參數解析
    2. inputs:
       - None
    3. return:
       - args: argparse.Namespace
    """
    ap = argparse.ArgumentParser(description="Fetch Fear & Greed index and upsample to 1D (UTC)")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--outfile", default=DEFAULT_FILENAME)
    return ap.parse_args()


def main() -> None:
    """
    1. 說明: 產生 1D FNG CSV + Feather
    2. inputs:
       - None (從 CLI 讀)
    3. return:
       - None
    """
    args = parse_args()
    df = load_fng_1d_utc(start_date=args.start, end_date=args.end)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / args.outfile
    df.to_csv(out_path, index=False)

    feather_path = out_path.with_suffix(".feather")
    df.to_feather(feather_path)

    print(f"Saved: {out_path.resolve()}")
    print(f"Saved: {feather_path.resolve()}")
    print(df.tail(3))


if __name__ == "__main__":
    main()
