from __future__ import annotations

"""VWAP 與價格代理特徵。"""

import math

from aggTrade.domain_types import AggTradeRecord
from aggTrade.trade_state import TradeBarState


class VwapPriceCalculator:
    """輸出 vwap/first/last/high/low/return。"""

    name = "vwap_prices"
    version = "1.0.0"

    def update(self, state: TradeBarState, trade: AggTradeRecord) -> None:
        # 價格序列由 BaseStatsCalculator 維護，這裡不需額外更新。
        _ = trade
        _ = state

    def finalize(self, state: TradeBarState) -> dict[str, float | int | bool]:
        vwap = math.nan
        if state.vwap_den > 0:
            vwap = state.vwap_num / state.vwap_den

        trade_return = math.nan
        if state.first_trade_price is not None and state.last_trade_price is not None and state.first_trade_price > 0:
            trade_return = state.last_trade_price / state.first_trade_price - 1.0

        return {
            "vwap": float(vwap),
            "first_trade_price": float(state.first_trade_price) if state.first_trade_price is not None else math.nan,
            "last_trade_price": float(state.last_trade_price) if state.last_trade_price is not None else math.nan,
            "trade_high_price": float(state.trade_high_price) if state.trade_high_price is not None else math.nan,
            "trade_low_price": float(state.trade_low_price) if state.trade_low_price is not None else math.nan,
            "trade_return": float(trade_return),
        }
