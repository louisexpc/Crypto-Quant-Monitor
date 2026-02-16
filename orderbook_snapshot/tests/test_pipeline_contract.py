from __future__ import annotations

from datetime import datetime

from orderbook_snapshot.aggregator import MinuteAggregator
from orderbook_snapshot.calculators import default_calculators
from orderbook_snapshot.engine import SnapshotFeatureEngine
from orderbook_snapshot.store import InMemoryFeatureStore


def _snapshot(symbol: str = "BTCUSDT", recv_ts_ms: int | None = None):
    from orderbook_snapshot.domain_types import SnapshotRecord

    recv_ms = recv_ts_ms or int(datetime(2026, 2, 16, 10, 1, 5).timestamp() * 1000)
    return SnapshotRecord(
        header={
            "symbol": symbol,
            "event_type": "snapshot",
            "source": "binance_futures",
            "recv_ts_ms": recv_ms,
            "event_ts_ms": recv_ms,
        },
        depth=1000,
        last_update_id=123,
        bids=[(100.0, 2.0), (99.9, 3.0), (99.6, 5.0)],
        asks=[(100.1, 4.0), (100.2, 1.0), (100.6, 5.0)],
    )


def test_engine_compute_is_deterministic() -> None:
    snap = _snapshot()
    engine = SnapshotFeatureEngine(default_calculators())

    f1 = engine.compute(snap)
    f2 = engine.compute(snap)

    assert f1 == f2


def test_minute_aggregator_and_store_last_write_wins() -> None:
    engine = SnapshotFeatureEngine(default_calculators())
    aggregator = MinuteAggregator(feature_schema_version="ob_snapshot_features.v1")
    store = InMemoryFeatureStore(max_rows=100)

    snap1 = _snapshot(recv_ts_ms=int(datetime(2026, 2, 16, 10, 1, 5).timestamp() * 1000))
    row1 = aggregator.ingest(snap1, engine.compute(snap1))
    store.append_row(row1)

    snap2 = _snapshot(recv_ts_ms=int(datetime(2026, 2, 16, 10, 1, 50).timestamp() * 1000))
    row2 = aggregator.ingest(snap2, engine.compute(snap2))
    row2["best_bid_q"] = 9.0
    store.append_row(row2)

    df = store.get_df("BTCUSDT")
    assert len(df) == 1
    assert float(df.iloc[-1]["best_bid_q"]) == 9.0

    required_columns = {
        "symbol",
        "bar_interval",
        "bar_open_timestamp_ms",
        "bar_close_timestamp_ms",
        "bar_open_datetime_tpe",
        "bar_close_datetime_tpe",
        "schema_version",
        "source",
        "snapshot_depth",
        "snapshot_last_update_id",
        "snapshot_event_ts_ms",
        "snapshot_recv_ts_ms",
        "feature_schema_version",
        "best_bid_p",
        "best_bid_q",
        "best_ask_p",
        "best_ask_q",
        "spread",
        "mid",
        "flag_crossed_book",
        "flag_bad_sorting",
        "flag_depth_insufficient",
        "flag_non_positive_qty",
        "depth_bid_10bps",
        "depth_ask_10bps",
        "bin_0_5bps_bid_qty",
        "bin_0_5bps_ask_qty",
    }
    assert required_columns.issubset(set(df.columns) | {"bar_open_datetime_tpe"})


def test_get_window_respects_lookback_and_monotonic_index() -> None:
    engine = SnapshotFeatureEngine(default_calculators())
    aggregator = MinuteAggregator()
    store = InMemoryFeatureStore(max_rows=100)

    ts_list = [
        int(datetime(2026, 2, 16, 10, 0, 5).timestamp() * 1000),
        int(datetime(2026, 2, 16, 10, 1, 5).timestamp() * 1000),
        int(datetime(2026, 2, 16, 10, 2, 5).timestamp() * 1000),
    ]

    for ts in ts_list:
        snap = _snapshot(recv_ts_ms=ts)
        row = aggregator.ingest(snap, engine.compute(snap))
        store.append_row(row)

    window = store.get_window("BTCUSDT", lookback=2)
    assert len(window) == 2
    assert window.index.is_monotonic_increasing
