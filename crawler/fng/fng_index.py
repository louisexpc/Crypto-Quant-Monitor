# crawler/fng/fng_index.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

DEFAULT_START = "2023-01-01"
DEFAULT_OUTDIR = Path("data/FNG")
DEFAULT_FILENAME = "fng_15m_utc.csv"


def load_fng_15m_utc(start_date: str = DEFAULT_START, end_date: str | None = None) -> pd.DataFrame:
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    j = requests.get(url, timeout=30).json()
    df = pd.DataFrame(j["data"])

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.loc[start_date:]
    if end_date:
        df = df.loc[:end_date]
    df["value"] = df["value"].astype(float)

    fng_15m = df[["value"]].resample("15min").ffill().rename(columns={"value": "fng"})
    fng_15m["fng_diff1"] = fng_15m["fng"].diff()
    roll = 24 * 7 * 4
    mean = fng_15m["fng"].rolling(roll, min_periods=24).mean()
    std = fng_15m["fng"].rolling(roll, min_periods=24).std()
    fng_15m["fng_z7d"] = (fng_15m["fng"] - mean) / (std + 1e-6)

    out = fng_15m.copy()
    out = out.loc[out.index >= pd.to_datetime(start_date, utc=True)]
    out["datetime"] = out.index.map(lambda ts: ts.isoformat())
    out["timestamp"] = (out.index.astype("int64") // 10**9).astype("int64")
    out = out[["datetime", "timestamp", "fng", "fng_diff1", "fng_z7d"]]
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch Fear & Greed index and upsample to 15m")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--outfile", default=DEFAULT_FILENAME)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = load_fng_15m_utc(start_date=args.start, end_date=args.end)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.outfile
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path.resolve()}")
    print(df.tail(3))


if __name__ == "__main__":
    main()
