"""多個 snapshot JSONL 檔案的 1m feature 聚合範例。

此範例示範如何：
1. 掃描 data/ 底下的多個 `snapshot_*.jsonl`
2. 解析成 SnapshotRecord
3. 依 timestamp 全域排序（避免同分鐘覆蓋順序錯誤）
4. 透過 module 聚合成 1m feature DataFrame
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from orderbook_snapshot import (
    InMemoryFeatureStore,
    MinuteAggregator,
    SnapshotFeatureEngine,
    default_calculators,
)
from orderbook_snapshot.adapters import JsonlSnapshotAdapter
from orderbook_snapshot.domain_types import SnapshotRecord


def _resolve_snapshot_ts_ms(snap: SnapshotRecord) -> int:
    """取出 snapshot 的分桶基準時間。

    Args:
        snap: 單筆 snapshot。

    Returns:
        依序採用 `event_ts_ms -> tx_ts_ms -> recv_ts_ms` 的毫秒時間。
    """
    header = snap.header
    if header.event_ts_ms is not None:
        return header.event_ts_ms
    if header.tx_ts_ms is not None:
        return header.tx_ts_ms
    return header.recv_ts_ms


def build_1m_feature_df_from_jsonl_dir(
    data_dir: str | Path,
    symbol: str = "BTCUSDT",
    max_rows: int = 200_000,
) -> pd.DataFrame:
    """從資料夾內多個 snapshot JSONL 產生 1m feature DataFrame。

    Args:
        data_dir: 存放 snapshot JSONL 的資料夾路徑。
        symbol: 要輸出的交易對。
        max_rows: store ring buffer 上限。

    Returns:
        指定 symbol 的 1m feature DataFrame（index 為 bar_open_datetime_tpe）。

    Raises:
        FileNotFoundError: 找不到任何 `snapshot_*.jsonl` 時拋出。
    """
    adapter = JsonlSnapshotAdapter(source="binance_futures")
    engine = SnapshotFeatureEngine(default_calculators())
    aggregator = MinuteAggregator(bar_interval="1m", feature_schema_version="ob_snapshot_features.v1")
    store = InMemoryFeatureStore(max_rows=max_rows)

    root = Path(data_dir)
    file_paths = sorted(root.rglob("snapshot_*.jsonl"))
    if not file_paths:
        raise FileNotFoundError(f"No snapshot jsonl files found under: {root}")

    # 全域排序（示範用）。
    # 若資料量很大，可改成檔名時間順序 + 檔內順序的 streaming 版本。
    records: list[tuple[int, int, str, SnapshotRecord]] = []
    seq = 0
    for path in file_paths:
        for snap in adapter.iter_file(path):
            if snap.header.symbol != symbol:
                continue
            ts_ms = _resolve_snapshot_ts_ms(snap)
            records.append((ts_ms, seq, str(path), snap))
            seq += 1

    records.sort(key=lambda x: (x[0], x[1]))

    for _, _, _, snap in records:
        features = engine.compute(snap)
        row = aggregator.ingest(snap, features)
        store.append_row(row)

    return store.get_df(symbol)


def main() -> None:
    """執行多檔 snapshot 聚合範例。"""
    # 依你專案結構，預設讀取 ./data
    data_dir = Path("data")
    symbol = "BTCUSDT"

    df = build_1m_feature_df_from_jsonl_dir(data_dir=data_dir, symbol=symbol, max_rows=200_000)

    print(f"[INFO] symbol={symbol}, rows={len(df)}, cols={len(df.columns)}")
    if not df.empty:
        print("\n[INFO] tail(3):")
        print(df.tail(3))
        print("\n[INFO] latest row:")
        print(df.iloc[-1])
    df.to_csv(f"{symbol}_1m_features.csv", index=True)


if __name__ == "__main__":
    main()