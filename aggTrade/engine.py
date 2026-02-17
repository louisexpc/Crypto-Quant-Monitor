from __future__ import annotations

"""Trade feature engine（update/finalize）。"""

from aggTrade.calculators.base import TradeFeatureCalculator
from aggTrade.domain_types import AggTradeRecord
from aggTrade.trade_state import TradeBarState


class TradeFeatureEngine:
    """負責對分鐘內狀態做增量更新與收斂輸出。"""

    def __init__(
        self,
        calculators: list[TradeFeatureCalculator],
        feature_schema_version: str = "trade_features.v1",
    ) -> None:
        self.calculators = calculators
        self.feature_schema_version = feature_schema_version

    def update(self, state: TradeBarState, trade: AggTradeRecord) -> None:
        """以單筆 trade 更新 state。"""
        for calculator in self.calculators:
            calculator.update(state, trade)

    def finalize(self, state: TradeBarState) -> dict[str, float | int | bool]:
        """把 state 收斂為特徵欄位字典。"""
        features: dict[str, float | int | bool] = {}
        for calculator in self.calculators:
            features.update(calculator.finalize(state))
        if state.extra_features:
            features.update(state.extra_features)
        return features
