from __future__ import annotations

"""JSONL snapshot adapter。"""

import json
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from orderbook_snapshot.domain_types import SnapshotRecord


class JsonlSnapshotAdapter:
    """將 JSONL 行資料轉為 `SnapshotRecord`。

    此 adapter 僅負責解析與型別正規化，不包含 feature 計算與儲存邏輯。
    """

    def __init__(self, source: str = "binance_futures") -> None:
        """初始化 adapter。

        Args:
            source: 寫入到 `EventRecordHeader.source` 的來源字串。
        """
        self.source = source

    @staticmethod
    def _to_ms(value: int | float | None) -> int | None:
        """把秒或毫秒 timestamp 正規化為毫秒。

        Args:
            value: 秒（float/int）或毫秒（int）timestamp。

        Returns:
            毫秒 timestamp；若輸入為 `None` 則回傳 `None`。
        """
        if value is None:
            return None
        if value > 1e12:
            return int(value)
        return int(float(value) * 1000)

    @staticmethod
    def _to_price_qty(levels: list[list[float | int]]) -> list[tuple[float, float]]:
        """把原始 depth level 轉為 `(price, qty)` tuple 清單。

        Args:
            levels: 二維陣列格式的 depth levels。

        Returns:
            tuple 型別的 `(price, qty)` 清單。
        """
        return [(float(price), float(qty)) for price, qty in levels]

    def parse_line(self, line: str) -> SnapshotRecord:
        """解析單行 JSON 字串為 `SnapshotRecord`。

        Args:
            line: JSONL 單行字串。

        Returns:
            轉換後的 `SnapshotRecord`。
        """
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
        """逐行讀取 JSONL 檔案並輸出 snapshot iterator。

        Args:
            file_path: JSONL 檔案路徑。

        Yields:
            每行成功解析後的 `SnapshotRecord`。

        Raises:
            ValueError: 當 JSON 格式錯誤或欄位不合法時拋出。
        """
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
