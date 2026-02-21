from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import pandas as pd

from orderbook_snapshot import (
    InMemoryFeatureStore,
    MinuteAggregator,
    SnapshotFeatureEngine,
    default_calculators,
)
from orderbook_snapshot.adapters import JsonlSnapshotAdapter
from orderbook_snapshot.domain_types import SnapshotRecord


# 只接受 snapshot_YYYYMMDDTHHMMSS.jsonl（會自然排除 snapshot_latest.jsonl）
_SNAPSHOT_RE = re.compile(r"^snapshot_(\d{8}T\d{6})\.jsonl$")


@dataclass(frozen=True)
class SnapshotFile:
    path: Path
    ts_tag: str           # YYYYMMDDTHHMMSS
    ts_key: datetime      # parsed datetime for sorting


def _resolve_snapshot_ts_ms(snap: SnapshotRecord) -> int:
    """
    取出 snapshot 的分桶基準時間（與你現有範例一致）。
    依序採用 event_ts_ms -> tx_ts_ms -> recv_ts_ms
    """
    header = snap.header
    if header.event_ts_ms is not None:
        return header.event_ts_ms
    if header.tx_ts_ms is not None:
        return header.tx_ts_ms
    return header.recv_ts_ms


def iter_snapshot_files(target_dir: Path) -> List[SnapshotFile]:
    """
    掃描 target_dir 底下所有 archive_*/snapshot_*.jsonl，並用檔名時間排序。
    """
    out: List[SnapshotFile] = []
    for p in target_dir.rglob("snapshot_*.jsonl"):
        m = _SNAPSHOT_RE.match(p.name)
        if not m:
            continue
        ts_tag = m.group(1)
        ts_key = datetime.strptime(ts_tag, "%Y%m%dT%H%M%S")
        out.append(SnapshotFile(path=p, ts_tag=ts_tag, ts_key=ts_key))

    out.sort(key=lambda x: x.ts_key)
    return out


def batched(items: List[SnapshotFile], batch_size: int) -> Iterator[List[SnapshotFile]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_df_from_files_streaming(
    files: List[SnapshotFile],
    *,
    symbol: str,
    source: str,
    max_rows: int,
) -> pd.DataFrame:
    """
    針對「一批檔案」做 streaming 聚合，不做全域 records 收集/排序（避免爆記憶體）。
    檔案順序使用檔名時間排序；檔內順序依 iter_file 產生順序。
    """
    adapter = JsonlSnapshotAdapter(source=source)
    engine = SnapshotFeatureEngine(default_calculators())
    aggregator = MinuteAggregator(bar_interval="1m", feature_schema_version="ob_snapshot_features.v1")
    store = InMemoryFeatureStore(max_rows=max_rows)

    # 若你擔心極端情況（檔內時間戳亂序），可在這裡做「小緩衝排序」，
    # 但通常 snapshot 檔本身已按時間寫入。
    for sf in files:
        for snap in adapter.iter_file(sf.path):
            if snap.header.symbol != symbol:
                continue
            features = engine.compute(snap)
            row = aggregator.ingest(snap, features)
            store.append_row(row)

    return store.get_df(symbol)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate snapshot_YYYYMMDDTHHMMSS.jsonl under target_dir in batches.")
    ap.add_argument("--target-dir", type=Path, required=True, help="target_dir path")
    ap.add_argument("--symbol", type=str, default="BTCUSDT", help="symbol to aggregate (e.g., BTCUSDT)")
    ap.add_argument("--source", type=str, default="binance_futures", help="adapter source")
    ap.add_argument("--batch-files", type=int, default=10_000, help="number of files per batch (e.g., 10000)")
    ap.add_argument("--max-rows", type=int, default=200_000, help="InMemoryFeatureStore ring buffer size")
    ap.add_argument("--overwrite", action="store_true", help="overwrite output csv if exists")
    args = ap.parse_args()

    target_dir: Path = args.target_dir
    if not target_dir.exists():
        raise FileNotFoundError(f"target-dir not found: {target_dir}")

    files = iter_snapshot_files(target_dir)
    if not files:
        raise FileNotFoundError(f"No snapshot_YYYYMMDDTHHMMSS.jsonl found under: {target_dir}")

    total = len(files)
    print(f"[INFO] Found {total} snapshot files under {target_dir}")

    for idx, batch in enumerate(batched(files, args.batch_files), start=1):
        start_tag = batch[0].ts_tag
        end_tag = batch[-1].ts_tag
        out_name = f"orderbook_snapshot_{start_tag}_{end_tag}.csv"
        out_path = target_dir / out_name

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] ({idx}) exists: {out_path.name}")
            continue

        print(f"[INFO] ({idx}) processing files={len(batch)} range={start_tag}..{end_tag}")

        df = build_df_from_files_streaming(
            batch,
            symbol=args.symbol,
            source=args.source,
            max_rows=args.max_rows,
        )

        # 你要的是「df 儲存在 target 資料夾路徑下」
        df.to_csv(out_path, index=True)
        print(f"[INFO] ({idx}) wrote rows={len(df)} cols={len(df.columns)} -> {out_path.name}")


if __name__ == "__main__":
    main()