from __future__ import annotations

"""基礎成交統計特徵（count/volume/notional）。"""

from aggTrade.domain_types import AggTradeRecord
from aggTrade.trade_state import TradeBarState


class BaseStatsCalculator:
    """更新與輸出基礎成交統計。"""

    name = "base_stats"
    version = "1.0.0"

    def update(self, state: TradeBarState, trade: AggTradeRecord) -> None:
        qty = float(trade.qty)
        price = float(trade.price)
        notional = price * qty

        state.trade_count += 1
        state.trade_volume += qty
        state.trade_notional += notional

        state.vwap_num += notional
        state.vwap_den += qty

        if state.first_trade_price is None:
            state.first_trade_price = price
        state.last_trade_price = price

        if state.trade_high_price is None or price > state.trade_high_price:
            state.trade_high_price = price
        if state.trade_low_price is None or price < state.trade_low_price:
            state.trade_low_price = price

        state.flag_no_trades = False

    def finalize(self, state: TradeBarState) -> dict[str, float | int | bool]:
        return {
            "trade_count": int(state.trade_count),
            "trade_volume": float(state.trade_volume),
            "trade_notional": float(state.trade_notional),
        }
