from __future__ import annotations

"""JSONL trade adapter。"""

import json
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from aggTrade.domain_types import AggTradeRecord


class JsonlTradeAdapter:
    """把 jsonl 行資料轉為 `AggTradeRecord`。"""

    def __init__(self, source: str = "binance_futures") -> None:
        self.source = source

    @staticmethod
    def _to_ms(value: int | float | str | None) -> int | None:
        if value is None:
            return None

        if isinstance(value, str):
            value = float(value)

        numeric = float(value)
        if numeric > 1e12:
            return int(numeric)
        return int(numeric * 1000)

    def parse_line(self, line: str) -> AggTradeRecord:
        payload = json.loads(line)
        msg = payload.get("msg", {})
        data = msg.get("data", {})

        symbol = str(data.get("s") or payload.get("symbol"))
        recv_ts_ms = self._to_ms(payload.get("recv_ts")) or 0
        event_ts_ms = self._to_ms(data.get("E"))
        tx_ts_ms = self._to_ms(data.get("T"))

        trade_ts_raw = data.get("T")
        trade_ts_ms = self._to_ms(trade_ts_raw) if trade_ts_raw is not None else 0

        qty_raw = data.get("q", data.get("nq", 0.0))

        return AggTradeRecord(
            header={
                "schema_version": "event.v1",
                "source": self.source,
                "symbol": symbol,
                "event_type": "trade",
                "event_ts_ms": event_ts_ms,
                "tx_ts_ms": tx_ts_ms,
                "recv_ts_ms": recv_ts_ms,
            },
            agg_trade_id=int(data.get("a", 0)),
            price=float(data.get("p", 0.0)),
            qty=float(qty_raw),
            trade_ts_ms=int(trade_ts_ms or 0),
            is_buyer_maker=bool(data.get("m", False)),
        )

    def iter_file(self, file_path: str | Path) -> Iterator[AggTradeRecord]:
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield self.parse_line(stripped)
                except (json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as err:
                    raise ValueError(f"Failed to parse trade jsonl at {path}:{line_no}: {err}") from err
