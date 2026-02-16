from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from orderbook_snapshot.domain_types import SnapshotRecord


class JsonlSnapshotAdapter:
    def __init__(self, source: str = "binance_futures") -> None:
        self.source = source

    @staticmethod
    def _to_ms(value: int | float | None) -> int | None:
        if value is None:
            return None
        if value > 1e12:
            return int(value)
        return int(float(value) * 1000)

    @staticmethod
    def _to_price_qty(levels: list[list[float | int]]) -> list[tuple[float, float]]:
        return [(float(price), float(qty)) for price, qty in levels]

    def parse_line(self, line: str) -> SnapshotRecord:
        payload = json.loads(line)

        symbol = str(payload["symbol"])
        snapshot_ts_ms = self._to_ms(payload.get("snapshot_ts"))
        recv_ts_ms = self._to_ms(payload.get("recv_ts", payload.get("snapshot_ts")))

        bids = self._to_price_qty(payload.get("bids", []))
        asks = self._to_price_qty(payload.get("asks", []))

        snapshot = SnapshotRecord(
            header={
                "schema_version": "event.v1",
                "source": self.source,
                "symbol": symbol,
                "event_type": "snapshot",
                "event_ts_ms": snapshot_ts_ms,
                "recv_ts_ms": recv_ts_ms or 0,
            },
            depth=int(payload.get("depth", min(len(bids), len(asks)))),
            last_update_id=payload.get("lastUpdateId"),
            bids=bids,
            asks=asks,
        )
        return snapshot

    def iter_file(self, file_path: str | Path) -> Iterator[SnapshotRecord]:
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield self.parse_line(stripped)
                except (json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as err:
                    raise ValueError(f"Failed to parse snapshot jsonl at {path}:{line_no}: {err}") from err
