from __future__ import annotations

"""預設 calculators 註冊入口。"""

from orderbook_snapshot.calculators.base import FeatureCalculator
from orderbook_snapshot.calculators.bps_bins import BpsBinsCalculator
from orderbook_snapshot.calculators.depth_bps import DepthWithinBpsCalculator
from orderbook_snapshot.calculators.quality import QualityFlagsCalculator
from orderbook_snapshot.calculators.tob import TopOfBookCalculator


def default_calculators() -> list[FeatureCalculator]:
    """建立 v1 預設 calculators 清單。

    Returns:
        依固定順序組合的 calculators。
    """
    return [
        TopOfBookCalculator(),
        QualityFlagsCalculator(),
        DepthWithinBpsCalculator(),
        BpsBinsCalculator(),
    ]


__all__ = [
    "FeatureCalculator",
    "TopOfBookCalculator",
    "QualityFlagsCalculator",
    "DepthWithinBpsCalculator",
    "BpsBinsCalculator",
    "default_calculators",
]
