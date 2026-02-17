from __future__ import annotations

"""方向性成交量特徵。"""

from aggTrade.domain_types import AggTradeRecord
from aggTrade.trade_state import TradeBarState


class DirectionalFlowCalculator:
    """依 `is_buyer_maker` 更新 buy/sell flow。"""

    name = "directional_flow"
    version = "1.0.0"

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def update(self, state: TradeBarState, trade: AggTradeRecord) -> None:
        qty = float(trade.qty)
        price = float(trade.price)
        notional = price * qty

        # Binance aggTrade: is_buyer_maker=True 通常代表主動賣。
        if trade.is_buyer_maker:
            state.sell_volume += qty
            state.sell_notional += notional
        else:
            state.buy_volume += qty
            state.buy_notional += notional

    def finalize(self, state: TradeBarState) -> dict[str, float | int | bool]:
        signed_volume = state.buy_volume - state.sell_volume
        volume_imbalance = signed_volume / (state.buy_volume + state.sell_volume + self.eps)

        return {
            "buy_volume": float(state.buy_volume),
            "sell_volume": float(state.sell_volume),
            "buy_notional": float(state.buy_notional),
            "sell_notional": float(state.sell_notional),
            "signed_volume": float(signed_volume),
            "volume_imbalance": float(volume_imbalance),
        }
