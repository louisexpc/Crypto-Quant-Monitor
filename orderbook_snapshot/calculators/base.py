from __future__ import annotations

"""Feature calculator 抽象協議定義。"""

from typing import Protocol

from orderbook_snapshot.domain_types import SnapshotRecord


class FeatureCalculator(Protocol):
    """特徵計算器協議。

    每個 calculator 接收一筆 `SnapshotRecord`，回傳一組 flattened features。
    """

    name: str
    version: str

    def compute(self, snap: SnapshotRecord) -> dict[str, float | int | bool]:
        """計算特徵。

        Args:
            snap: 單筆 snapshot。

        Returns:
            特徵字典。
        """
        ...
