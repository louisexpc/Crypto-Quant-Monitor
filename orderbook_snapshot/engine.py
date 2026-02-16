from __future__ import annotations

from orderbook_snapshot.calculators.base import FeatureCalculator
from orderbook_snapshot.domain_types import SnapshotRecord


class SnapshotFeatureEngine:
    """將多個 FeatureCalculator 組合成單一計算引擎。

    此引擎只負責 `snapshot -> features dict` 的純計算，不處理時間分桶、
    DataFrame 儲存或任何 I/O。
    """

    def __init__(
        self,
        calculators: list[FeatureCalculator],
        feature_schema_version: str = "ob_snapshot_features.v1",
    ) -> None:
        """初始化特徵計算引擎。

        Args:
            calculators: 依序執行的特徵計算器清單。
            feature_schema_version: 目前特徵集合版本字串。
        """
        self.calculators = calculators
        self.feature_schema_version = feature_schema_version

    def compute(self, snap: SnapshotRecord) -> dict[str, float | int | bool]:
        """計算單筆 snapshot 的特徵字典。

        若 calculators 產生重複 key，後執行者會覆蓋先執行者（last key wins）。

        Args:
            snap: 已完成驗證的 SnapshotRecord。

        Returns:
            合併後的特徵字典，value 型別為 float/int/bool。
        """
        features: dict[str, float | int | bool] = {}
        for calculator in self.calculators:
            features.update(calculator.compute(snap))
        return features
