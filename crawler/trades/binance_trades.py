# crawler/trades/binance_trades.py
"""Download Binance trade archives via CLI (resume-friendly) and merge to CSV."""

from __future__ import annotations

import argparse
import glob
import os
import re
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision"
PREFIX_TEMPLATE = r"data/futures/um/daily/trades/{symbol}"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_OUTDIR = Path("data/binance_trades") / DEFAULT_SYMBOL
DEFAULT_START = "2023-01-01"

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _zip_name(symbol: str, day: date) -> str:
    return f"{symbol}-trades-{day.isoformat()}.zip"


def dl_one(day: date, symbol: str, outdir: Path, base_url: str = BASE_URL):
    """Download one daily archive if missing; return file path and flag."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = PREFIX_TEMPLATE.format(symbol=symbol)
    fname = _zip_name(symbol, day)
    url = f"{base_url}/{prefix}/{fname}"
    dst = outdir / fname
    if dst.exists():
        return dst, False

    r = requests.get(url, timeout=60)
    if r.status_code == 200:
        with open(dst, "wb") as f:
            f.write(r.content)
        return dst, True
    return None, False


def unzip_csvs(zpath: Path):
    """Yield DataFrames from CSV members inside a zip file."""
    with zipfile.ZipFile(zpath, "r") as z:
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            with z.open(name) as f:
                yield pd.read_csv(f)


def _existing_dates(outdir: Path, symbol: str) -> list[date]:
    pattern = outdir / f"{symbol}-trades-*.zip"
    dates: list[date] = []
    for path in glob.glob(str(pattern)):
        name = os.path.basename(path)
        m = DATE_RE.search(name)
        if not m:
            continue
        try:
            dates.append(date.fromisoformat(m.group(1)))
        except ValueError:
            continue
    return sorted(set(dates))


def _first_missing(dates: list[date], start: date) -> date:
    cur = start
    for d in dates:
        if d < cur:
            continue
        if d == cur:
            cur = d + timedelta(days=1)
        else:
            break
    return cur


def _zip_path(outdir: Path, symbol: str, day: date) -> Path:
    return outdir / _zip_name(symbol, day)


def _concat_frames(outdir: Path, symbol: str, start: date, end: date) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cur = start
    while cur <= end:
        zpath = _zip_path(outdir, symbol, cur)
        if zpath.exists():
            for df in unzip_csvs(zpath):
                frames.append(df)
        else:
            print(f"[warn] missing archive for {cur}, skip in concat")
        cur += timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if "time" in merged.columns:
        merged.sort_values("time", inplace=True)
    return merged


def backfill(
    start: str = DEFAULT_START,
    end: str | None = None,
    outdir: str | Path = DEFAULT_OUTDIR,
    symbol: str = DEFAULT_SYMBOL,
    base_url: str = BASE_URL,
    output_csv: str | None = None,
    resume: bool = True,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    start_date = date.fromisoformat(start)
    end_date = date.today() if end is None else date.fromisoformat(end)
    if end_date < start_date:
        raise ValueError("end date must be >= start date")

    resume_start = _first_missing(_existing_dates(outdir, symbol), start_date) if resume else start_date
    cur = resume_start
    while cur <= end_date:
        z, is_new = dl_one(cur, symbol=symbol, outdir=outdir, base_url=base_url)
        status = "downloaded" if is_new else "cached"
        if z:
            print(f"[{cur}] {status}: {z}")
        else:
            print(f"[{cur}] missing on server")
        cur += timedelta(days=1)

    merged = _concat_frames(outdir, symbol, start_date, end_date)
    if output_csv is None:
        stamp = start_date.strftime("%Y%m%d")
        output_csv = outdir / f"{symbol}_trades_{stamp}on.csv"
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Saved {len(merged)} rows → {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download Binance UM daily trades and merge into one CSV")
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    ap.add_argument("--output_csv", default=None, help="Optional merged CSV path")
    ap.add_argument("--base_url", default=BASE_URL)
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    backfill(
        start=args.start,
        end=args.end,
        outdir=args.outdir,
        symbol=args.symbol.upper(),
        base_url=args.base_url,
        output_csv=args.output_csv,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
