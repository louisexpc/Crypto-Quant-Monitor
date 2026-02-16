from __future__ import annotations

from orderbook_snapshot.calculators.base import FeatureCalculator
from orderbook_snapshot.calculators.bps_bins import BpsBinsCalculator
from orderbook_snapshot.calculators.depth_bps import DepthWithinBpsCalculator
from orderbook_snapshot.calculators.quality import QualityFlagsCalculator
from orderbook_snapshot.calculators.tob import TopOfBookCalculator


def default_calculators() -> list[FeatureCalculator]:
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
