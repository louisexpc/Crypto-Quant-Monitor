"""Trade 事件驅動分鐘聚合器。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from aggTrade.calculators import default_calculators
from aggTrade.domain_types import (
    AggTradeRecord,
    BarInterval,
    TZ_TPE,
    floor_to_bar_open_in_tz,
    make_bar_header,
)
from aggTrade.engine import TradeFeatureEngine
from aggTrade.trade_state import TradeBarState


class TradeMinuteAggregator:
    """將逐筆 trade 累積成 1m（或指定 interval）bar row。"""

    def __init__(
        self,
        engine: TradeFeatureEngine | None = None,
        bar_interval: BarInterval = "1m",
        source: str = "binance_futures",
        feature_schema_version: str = "trade_features.v1",
    ) -> None:
        self.engine = engine or TradeFeatureEngine(default_calculators(), feature_schema_version=feature_schema_version)
        self.bar_interval = bar_interval
        self.source = source
        self.feature_schema_version = feature_schema_version
        self._states: dict[str, TradeBarState] = {}
        self._latest_finalized_bar_open_ms: dict[str, int] = {}

    def _resolve_trade_ts_ms(self, trade: AggTradeRecord) -> tuple[int, bool]:
        """解析 trade 分桶時間，並標記是否使用 fallback。"""
        if trade.trade_ts_ms > 0:
            return trade.trade_ts_ms, False

        for candidate in (trade.header.tx_ts_ms, trade.header.event_ts_ms, trade.header.recv_ts_ms):
            if candidate is not None:
                return int(candidate), True

        return 0, True

    def _new_state(self, symbol: str, bar_open_ts_ms: int, flag_missing_trade_ts: bool) -> TradeBarState:
        state = TradeBarState(symbol=symbol, bar_open_timestamp_ms=bar_open_ts_ms, bar_interval=self.bar_interval)
        if flag_missing_trade_ts:
            state.flag_missing_trade_ts = True
        return state

    def _build_row(self, state: TradeBarState) -> dict[str, Any]:
        features = self.engine.finalize(state)

        bar_header = make_bar_header(
            symbol=state.symbol,
            bar_interval=state.bar_interval,
            bar_open_timestamp_ms=state.bar_open_timestamp_ms,
            source=self.source,
            schema_version="barframe.v1",
            tz=TZ_TPE,
        )

        row: dict[str, Any] = {
            "symbol": bar_header.symbol,
            "bar_interval": bar_header.bar_interval,
            "bar_open_timestamp_ms": bar_header.bar_open_timestamp_ms,
            "bar_close_timestamp_ms": bar_header.bar_close_timestamp_ms,
            "bar_open_datetime_tpe": bar_header.bar_open_datetime_tpe,
            "bar_close_datetime_tpe": bar_header.bar_close_datetime_tpe,
            "schema_version": bar_header.schema_version,
            "source": bar_header.source,
            "feature_schema_version": self.feature_schema_version,
            "last_trade_recv_ts_ms": int(state.last_trade_recv_ts_ms),
        }
        row.update(features)

        if not isinstance(row["bar_open_datetime_tpe"], datetime) or row["bar_open_datetime_tpe"].tzinfo is None:
            raise ValueError("bar_open_datetime_tpe must be timezone-aware datetime")

        return row

    def ingest(self, trade: AggTradeRecord) -> list[dict[str, Any]]:
        """輸入單筆 trade，必要時 finalize 前一分鐘並回傳 row。"""
        symbol = trade.header.symbol
        trade_ts_ms, flag_missing_trade_ts = self._resolve_trade_ts_ms(trade)
        if trade_ts_ms <= 0:
            # 無法解析有效時間戳的 trade：不分桶，不更新引擎，僅標記現有狀態。
            state = self._states.get(symbol)
            if state is not None and flag_missing_trade_ts:
                state.flag_missing_trade_ts = True
            return []
        bar_open_ts_ms = floor_to_bar_open_in_tz(trade_ts_ms, interval=self.bar_interval, tz=TZ_TPE)

        state = self._states.get(symbol)
        if state is None:
            new_state = self._new_state(symbol=symbol, bar_open_ts_ms=bar_open_ts_ms, flag_missing_trade_ts=flag_missing_trade_ts)
            self._states[symbol] = new_state
            self.engine.update(new_state, trade)
            return []

        if bar_open_ts_ms == state.bar_open_timestamp_ms:
            if flag_missing_trade_ts:
                state.flag_missing_trade_ts = True
            self.engine.update(state, trade)
            return []

        if bar_open_ts_ms > state.bar_open_timestamp_ms:
            finalized = self._build_row(state)
            self._latest_finalized_bar_open_ms[symbol] = state.bar_open_timestamp_ms

            new_state = self._new_state(symbol=symbol, bar_open_ts_ms=bar_open_ts_ms, flag_missing_trade_ts=flag_missing_trade_ts)
            self._states[symbol] = new_state
            self.engine.update(new_state, trade)
            return [finalized]

        # bar_open_ts_ms < current bar: 視為 late trade（v1 不回補）
        state.late_trade_count += 1
        state.flag_out_of_order_ts = True
        if flag_missing_trade_ts:
            state.flag_missing_trade_ts = True
        return []

    def flush(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """強制 finalize 目前開啟中的 bar。"""
        if symbol is None:
            symbols = sorted(self._states.keys())
        else:
            symbols = [symbol] if symbol in self._states else []

        rows: list[dict[str, Any]] = []
        for sym in symbols:
            state = self._states.pop(sym)
            rows.append(self._build_row(state))
            self._latest_finalized_bar_open_ms[sym] = state.bar_open_timestamp_ms

        return rows
