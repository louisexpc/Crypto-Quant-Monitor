from __future__ import annotations

"""Trade 品質旗標特徵。"""

from aggTrade.domain_types import AggTradeRecord
from aggTrade.trade_state import TradeBarState


class TradeQualityCalculator:
    """更新與輸出 quality flags。"""

    name = "trade_quality"
    version = "1.0.0"

    def update(self, state: TradeBarState, trade: AggTradeRecord) -> None:
        if trade.qty <= 0:
            state.flag_non_positive_qty = True
        if trade.price <= 0:
            state.flag_non_positive_price = True

        if state.last_trade_ts_ms is not None and trade.trade_ts_ms > 0 and trade.trade_ts_ms < state.last_trade_ts_ms:
            state.flag_out_of_order_ts = True

        if trade.trade_ts_ms > 0:
            state.last_trade_ts_ms = trade.trade_ts_ms

        state.last_trade_recv_ts_ms = max(state.last_trade_recv_ts_ms, trade.header.recv_ts_ms)

    def finalize(self, state: TradeBarState) -> dict[str, float | int | bool]:
        return {
            "flag_no_trades": bool(state.flag_no_trades),
            "flag_non_positive_qty": bool(state.flag_non_positive_qty),
            "flag_non_positive_price": bool(state.flag_non_positive_price),
            "flag_out_of_order_ts": bool(state.flag_out_of_order_ts),
            "flag_missing_trade_ts": bool(state.flag_missing_trade_ts),
            "late_trade_count": int(state.late_trade_count),
        }
