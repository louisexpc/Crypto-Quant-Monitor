from __future__ import annotations

from datetime import datetime

from orderbook_snapshot.domain_types import TZ_TPE, floor_to_bar_open_in_tz, make_bar_header


def test_floor_to_bar_open_in_tz_aligns_to_tpe_minute_boundary() -> None:
    ts_ms = int(datetime(2026, 2, 16, 10, 1, 23, 456000, tzinfo=TZ_TPE).timestamp() * 1000)

    open_ms = floor_to_bar_open_in_tz(ts_ms, interval="1m", tz=TZ_TPE)
    open_dt = datetime.fromtimestamp(open_ms / 1000.0, tz=TZ_TPE)

    assert open_dt.second == 0
    assert open_dt.microsecond == 0
    assert open_dt.minute == 1


def test_make_bar_header_close_ts_is_open_plus_60s_minus_1ms() -> None:
    open_ms = int(datetime(2026, 2, 16, 10, 1, 0, tzinfo=TZ_TPE).timestamp() * 1000)

    header = make_bar_header(
        symbol="BTCUSDT",
        bar_interval="1m",
        bar_open_timestamp_ms=open_ms,
        tz=TZ_TPE,
    )

    assert header.bar_close_timestamp_ms == open_ms + 60_000 - 1
