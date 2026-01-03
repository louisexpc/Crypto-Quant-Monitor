"""Unzip all Binance trade archives and merge them into a single CSV."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
from pathlib import Path
from zipfile import ZipFile

DEFAULT_SYMBOL = "BTCUSDT"

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _extract_date(path: Path) -> str | None:
    m = DATE_RE.search(path.name)
    return m.group(1) if m else None


def _pick_csv_name(zf: ZipFile) -> str:
    for name in zf.namelist():
        lower = name.lower()
        if lower.endswith(".csv") or lower.endswith(".csv.gz"):
            return name
    raise FileNotFoundError("No CSV/CSV.GZ found inside archive")


def _append_zip(zpath: Path, writer: csv.writer, header_ref: list[str] | None) -> list[str] | None:
    with ZipFile(zpath) as zf:
        csv_name = _pick_csv_name(zf)
        with zf.open(csv_name) as raw:
            stream = gzip.GzipFile(fileobj=raw) if csv_name.lower().endswith(".gz") else raw
            text = io.TextIOWrapper(stream, encoding="utf-8")
            reader = csv.reader(text)
            file_header = next(reader, None)
            if file_header:
                if header_ref is None:
                    writer.writerow(file_header)
                    header_ref = file_header
                elif header_ref != file_header:
                    raise ValueError(f"Header mismatch in {zpath.name}")
            for row in reader:
                writer.writerow(row)
    return header_ref


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unzip Binance daily trades and merge to CSV")
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Symbol (e.g., BTCUSDT)")
    ap.add_argument(
        "--input_dir",
        default=None,
        help="Directory containing daily trade zips (default: data/binance_trades/<symbol>)",
    )
    ap.add_argument(
        "--output_csv",
        default=None,
        help="Output merged CSV path (default: <input_dir>/<symbol>_trades_merged.csv)",
    )
    return ap.parse_args()


def main(args: argparse.Namespace | None = None) -> Path:
    args = args or parse_args()
    symbol = args.symbol.upper()
    input_dir = Path(args.input_dir) if args.input_dir else Path("data/binance_trades") / symbol
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    pattern = f"{symbol}-trades-*.zip"
    zips = sorted(
        input_dir.glob(pattern),
        key=lambda p: (_extract_date(p) is None, _extract_date(p) or p.name),
    )
    if not zips:
        raise FileNotFoundError(f"No zip files matched {pattern} under {input_dir}")

    out_path = Path(args.output_csv) if args.output_csv else input_dir / f"{symbol}_trades_merged.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header: list[str] | None = None
    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        for idx, zpath in enumerate(zips, 1):
            print(f"[{idx}/{len(zips)}] Merging {zpath.name} ...")
            header = _append_zip(zpath, writer, header)

    print(f"Done. Combined {len(zips)} archives into {out_path}")
    return out_path


if __name__ == "__main__":
    main()

