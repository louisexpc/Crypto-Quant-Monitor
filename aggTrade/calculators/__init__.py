from __future__ import annotations

"""Trade calculators 註冊入口。"""

from aggTrade.calculators.base import TradeFeatureCalculator
from aggTrade.calculators.base_stats import BaseStatsCalculator
from aggTrade.calculators.directional_flow import DirectionalFlowCalculator
from aggTrade.calculators.quality import TradeQualityCalculator
from aggTrade.calculators.vwap_prices import VwapPriceCalculator


def default_calculators() -> list[TradeFeatureCalculator]:
    """建立 v1 預設 calculators。"""
    return [
        BaseStatsCalculator(),
        DirectionalFlowCalculator(),
        VwapPriceCalculator(),
        TradeQualityCalculator(),
    ]


__all__ = [
    "TradeFeatureCalculator",
    "BaseStatsCalculator",
    "DirectionalFlowCalculator",
    "VwapPriceCalculator",
    "TradeQualityCalculator",
    "default_calculators",
]
