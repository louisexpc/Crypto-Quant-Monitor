from __future__ import annotations

from orderbook_snapshot.calculators.base import FeatureCalculator
from orderbook_snapshot.domain_types import SnapshotRecord


class SnapshotFeatureEngine:
    def __init__(
        self,
        calculators: list[FeatureCalculator],
        feature_schema_version: str = "ob_snapshot_features.v1",
    ) -> None:
        self.calculators = calculators
        self.feature_schema_version = feature_schema_version

    def compute(self, snap: SnapshotRecord) -> dict[str, float | int | bool]:
        features: dict[str, float | int | bool] = {}
        for calculator in self.calculators:
            features.update(calculator.compute(snap))
        return features
