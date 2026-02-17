from __future__ import annotations

from datetime import datetime

from aggTrade.aggregator import TradeMinuteAggregator
from aggTrade.calculators import default_calculators
from aggTrade.domain_types import TZ_TPE, AggTradeRecord, floor_to_bar_open_in_tz
from aggTrade.engine import TradeFeatureEngine
from aggTrade.store import InMemoryFeatureStore


def _ts_ms(y: int, m: int, d: int, hh: int, mm: int, ss: int) -> int:
    return int(datetime(y, m, d, hh, mm, ss, tzinfo=TZ_TPE).timestamp() * 1000)


def _trade(*, ts_ms: int, price: float, qty: float, is_buyer_maker: bool, recv_ts_ms: int | None = None) -> AggTradeRecord:
    recv = recv_ts_ms or ts_ms + 5
    return AggTradeRecord(
        header={
            "symbol": "BTCUSDT",
            "event_type": "trade",
            "source": "binance_futures",
            "event_ts_ms": ts_ms,
            "tx_ts_ms": ts_ms,
            "recv_ts_ms": recv,
        },
        agg_trade_id=1,
        price=price,
        qty=qty,
        trade_ts_ms=ts_ms,
        is_buyer_maker=is_buyer_maker,
    )


def test_rollover_finalize_once_on_minute_change() -> None:
    agg = TradeMinuteAggregator()

    t1 = _trade(ts_ms=_ts_ms(2026, 2, 17, 10, 0, 5), price=100.0, qty=1.0, is_buyer_maker=False)
    t2 = _trade(ts_ms=_ts_ms(2026, 2, 17, 10, 0, 30), price=101.0, qty=2.0, is_buyer_maker=True)
    t3 = _trade(ts_ms=_ts_ms(2026, 2, 17, 10, 1, 1), price=102.0, qty=1.0, is_buyer_maker=False)

    assert agg.ingest(t1) == []
    assert agg.ingest(t2) == []

    rows = agg.ingest(t3)
    assert len(rows) == 1
    row = rows[0]

    expected_open = floor_to_bar_open_in_tz(t1.trade_ts_ms, interval="1m", tz=TZ_TPE)
    assert row["bar_open_timestamp_ms"] == expected_open
    assert int(row["trade_count"]) == 2


def test_vwap_and_directional_features() -> None:
    agg = TradeMinuteAggregator()

    t1 = _trade(ts_ms=_ts_ms(2026, 2, 17, 11, 0, 1), price=100.0, qty=2.0, is_buyer_maker=False)
    t2 = _trade(ts_ms=_ts_ms(2026, 2, 17, 11, 0, 20), price=110.0, qty=1.0, is_buyer_maker=True)

    agg.ingest(t1)
    agg.ingest(t2)
    rows = agg.flush("BTCUSDT")

    assert len(rows) == 1
    row = rows[0]

    assert abs(float(row["vwap"]) - ((100.0 * 2.0 + 110.0 * 1.0) / 3.0)) < 1e-12
    assert abs(float(row["buy_volume"]) - 2.0) < 1e-12
    assert abs(float(row["sell_volume"]) - 1.0) < 1e-12
    assert abs(float(row["signed_volume"]) - 1.0) < 1e-12
    assert abs(float(row["volume_imbalance"]) - (1.0 / 3.0)) < 1e-12
    assert abs(float(row["trade_return"]) - 0.1) < 1e-12


def test_store_monotonic_index_and_no_duplicate_minute() -> None:
    agg = TradeMinuteAggregator(engine=TradeFeatureEngine(default_calculators()), bar_interval="1m")
    store = InMemoryFeatureStore(max_rows=100)

    trades = [
        _trade(ts_ms=_ts_ms(2026, 2, 17, 12, 0, 5), price=100.0, qty=1.0, is_buyer_maker=False),
        _trade(ts_ms=_ts_ms(2026, 2, 17, 12, 0, 45), price=101.0, qty=1.0, is_buyer_maker=False),
        _trade(ts_ms=_ts_ms(2026, 2, 17, 12, 1, 5), price=102.0, qty=1.0, is_buyer_maker=True),
        _trade(ts_ms=_ts_ms(2026, 2, 17, 12, 2, 5), price=103.0, qty=1.0, is_buyer_maker=True),
    ]

    for trade in trades:
        rows = agg.ingest(trade)
        for row in rows:
            store.append_row(row)

    for row in agg.flush("BTCUSDT"):
        store.append_row(row)

    df = store.get_df("BTCUSDT")
    assert df.index.is_monotonic_increasing
    assert int(df["bar_open_timestamp_ms"].duplicated().sum()) == 0
