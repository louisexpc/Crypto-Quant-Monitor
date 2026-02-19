from __future__ import annotations

"""Trade 1m 聚合時的 state 容器。"""

from dataclasses import dataclass, field

from aggTrade.domain_types import BarInterval


@dataclass(slots=True)
class TradeBarState:
    """單一 symbol、單一 bar 的累積狀態。"""

    symbol: str
    bar_open_timestamp_ms: int
    bar_interval: BarInterval = "1m"

    trade_count: int = 0
    trade_volume: float = 0.0
    trade_notional: float = 0.0

    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_notional: float = 0.0
    sell_notional: float = 0.0

    vwap_num: float = 0.0
    vwap_den: float = 0.0

    first_trade_price: float | None = None
    last_trade_price: float | None = None
    trade_high_price: float | None = None
    trade_low_price: float | None = None

    flag_non_positive_qty: bool = False
    flag_non_positive_price: bool = False
    flag_out_of_order_ts: bool = False
    flag_missing_trade_ts: bool = False
    flag_no_trades: bool = True

    late_trade_count: int = 0
    last_trade_ts_ms: int | None = None
    last_trade_recv_ts_ms: int = 0

    extra_features: dict[str, float | int | bool] = field(default_factory=dict)
