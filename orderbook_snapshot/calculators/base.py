from __future__ import annotations

from typing import Protocol

from orderbook_snapshot.domain_types import SnapshotRecord


class FeatureCalculator(Protocol):
    name: str
    version: str

    def compute(self, snap: SnapshotRecord) -> dict[str, float | int | bool]:
        ...
