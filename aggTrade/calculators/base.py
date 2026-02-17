from __future__ import annotations

"""Trade feature calculator 協議。"""

from typing import Protocol

from aggTrade.domain_types import AggTradeRecord
from aggTrade.trade_state import TradeBarState


class TradeFeatureCalculator(Protocol):
    """Trade 特徵計算器協議（update/finalize）。"""

    name: str
    version: str

    def update(self, state: TradeBarState, trade: AggTradeRecord) -> None:
        """以單筆 trade 更新狀態。"""

    def finalize(self, state: TradeBarState) -> dict[str, float | int | bool]:
        """把狀態收斂為輸出特徵欄位。"""
