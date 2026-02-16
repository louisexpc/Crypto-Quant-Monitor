from __future__ import annotations

from datetime import datetime
from typing import Any

from orderbook_snapshot.domain_types import (
    TZ_TPE,
    BarInterval,
    SnapshotRecord,
    floor_to_bar_open_in_tz,
    make_bar_header,
)


class MinuteAggregator:
    def __init__(
        self,
        bar_interval: BarInterval = "1m",
        source: str = "binance_futures",
        feature_schema_version: str = "ob_snapshot_features.v1",
    ) -> None:
        self.bar_interval = bar_interval
        self.source = source
        self.feature_schema_version = feature_schema_version

    def _resolve_bucket_ts_ms(self, snap: SnapshotRecord) -> int:
        header = snap.header
        if header.event_ts_ms is not None:
            return header.event_ts_ms
        if header.tx_ts_ms is not None:
            return header.tx_ts_ms
        return header.recv_ts_ms

    def ingest(self, snap: SnapshotRecord, features: dict[str, float | int | bool]) -> dict[str, Any]:
        bar_open_ts_ms = floor_to_bar_open_in_tz(
            self._resolve_bucket_ts_ms(snap),
            interval=self.bar_interval,
            tz=TZ_TPE,
        )
        bar_header = make_bar_header(
            symbol=snap.header.symbol,
            bar_interval=self.bar_interval,
            bar_open_timestamp_ms=bar_open_ts_ms,
            source=self.source,
            schema_version="barframe.v1",
            tz=TZ_TPE,
        )

        required_tob_fields = {
            "best_bid_p",
            "best_bid_q",
            "best_ask_p",
            "best_ask_q",
            "spread",
            "mid",
        }
        missing = required_tob_fields - set(features)
        if missing:
            raise ValueError(f"Missing required TOB fields from features: {sorted(missing)}")

        row: dict[str, Any] = {
            "symbol": bar_header.symbol,
            "bar_interval": bar_header.bar_interval,
            "bar_open_timestamp_ms": bar_header.bar_open_timestamp_ms,
            "bar_close_timestamp_ms": bar_header.bar_close_timestamp_ms,
            "bar_open_datetime_tpe": bar_header.bar_open_datetime_tpe,
            "bar_close_datetime_tpe": bar_header.bar_close_datetime_tpe,
            "schema_version": bar_header.schema_version,
            "source": bar_header.source,
            "snapshot_depth": snap.depth,
            "snapshot_last_update_id": snap.last_update_id,
            "snapshot_event_ts_ms": snap.header.event_ts_ms,
            "snapshot_recv_ts_ms": snap.header.recv_ts_ms,
            "feature_schema_version": self.feature_schema_version,
        }
        row.update(features)

        if not isinstance(row["bar_open_datetime_tpe"], datetime) or row["bar_open_datetime_tpe"].tzinfo is None:
            raise ValueError("bar_open_datetime_tpe must be timezone-aware datetime")

        return row
