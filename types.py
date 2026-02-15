# types.py
"""
全資料集型定義與工具函式。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Literal, Sequence, Tuple

from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =========================
# Time / Interval utilities
# =========================

TZ_UTC = timezone.utc
TZ_TPE = ZoneInfo("Asia/Taipei")


BarInterval = Literal[
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
]


_INTERVAL_TO_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 3 * 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "6h": 6 * 60 * 60,
    "8h": 8 * 60 * 60,
    "12h": 12 * 60 * 60,
    "1d": 24 * 60 * 60,
    "3d": 3 * 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
    # "1M" is calendar-month based; see floor_to_bar_open_in_tz for handling.
}


def datetime_from_ts_ms(ts_ms: int, tz: timezone | ZoneInfo) -> datetime:
    """Convert epoch milliseconds to timezone-aware datetime.
    把 epoch ms → timezone-aware datetime（先當 UTC 再轉 tz）
    """
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=TZ_UTC).astimezone(tz)


def ts_ms_from_datetime(dt: datetime) -> int:
    """Convert timezone-aware datetime to epoch milliseconds (UTC).
    把 timezone-aware datetime → epoch ms（UTC 基準）
    """
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return int(dt.astimezone(TZ_UTC).timestamp() * 1000)


def floor_to_bar_open_in_tz(ts_ms: int, interval: BarInterval, tz: ZoneInfo = TZ_TPE) -> int:
    """
    Floor a timestamp to the bar open timestamp (ms) aligned in a given timezone.

    - Uses `tz` alignment (default: Asia/Taipei) to match your OHLCV indexing convention.
    - Returns epoch ms (UTC-based), suitable as `bar_open_timestamp_ms`.
    把任意時間戳「對齊到該 interval 的 bar open」
    - 用 Asia/Taipei 對齊
    - 回傳的是 bar open 的 epoch ms（供你當 canonical key）
    """
    dt_local = datetime_from_ts_ms(ts_ms, tz=tz)

    if interval == "1M":
        # Floor to first day of month at 00:00:00 in tz
        floored = dt_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return ts_ms_from_datetime(floored)

    if interval == "1w":
        # Floor to Monday 00:00:00 in tz (ISO week start)
        # If you prefer Sunday start, change weekday logic here.
        days_to_monday = dt_local.weekday()  # Monday=0
        floored_date = (dt_local - timedelta(days=days_to_monday)).date()
        floored = datetime(
            floored_date.year, floored_date.month, floored_date.day, 0, 0, 0, 0, tzinfo=tz
        )
        return ts_ms_from_datetime(floored)

    if interval in ("1d", "3d"):
        # Floor to local midnight; for 3d, floor to blocks starting at an epoch-aligned day boundary in tz.
        base = dt_local.replace(hour=0, minute=0, second=0, microsecond=0)
        if interval == "3d":
            # Define an epoch anchor in tz to create stable 3-day blocks:
            # anchor = 1970-01-01 00:00:00 in tz
            anchor = datetime(1970, 1, 1, 0, 0, 0, 0, tzinfo=tz)
            delta_days = (base.date() - anchor.date()).days
            block_days = (delta_days // 3) * 3
            floored = anchor + timedelta(days=block_days)
            return ts_ms_from_datetime(floored)
        return ts_ms_from_datetime(base)

    # Minute/hour intervals: fixed seconds
    seconds = _INTERVAL_TO_SECONDS.get(interval)
    if seconds is None:
        raise ValueError(f"Unsupported interval: {interval}")

    # Floor in local time by converting to seconds since local epoch-ish anchor.
    # Use local timestamp seconds for alignment.
    local_epoch_seconds = int(dt_local.timestamp())
    floored_seconds = (local_epoch_seconds // seconds) * seconds
    floored_local = datetime.fromtimestamp(floored_seconds, tz=tz)
    return ts_ms_from_datetime(floored_local)


def bar_close_ts_ms(bar_open_ts_ms: int, interval: BarInterval, tz: ZoneInfo = TZ_TPE) -> int:
    """
    Compute bar close timestamp (ms) as open + interval duration.
    For calendar month ("1M"), close is next month's first day at 00:00:00 in tz minus 1 ms.
    給定 bar open ms，算 bar close ms（通常 = open + duration - 1ms）
    """
    open_local = datetime_from_ts_ms(bar_open_ts_ms, tz=tz)

    if interval == "1M":
        year = open_local.year
        month = open_local.month
        if month == 12:
            nxt = open_local.replace(year=year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            nxt = open_local.replace(month=month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return ts_ms_from_datetime(nxt) - 1

    if interval == "1w":
        nxt = open_local + timedelta(days=7)
        nxt = nxt.replace(hour=0, minute=0, second=0, microsecond=0)
        return ts_ms_from_datetime(nxt) - 1

    if interval == "3d":
        nxt = open_local + timedelta(days=3)
        nxt = nxt.replace(hour=0, minute=0, second=0, microsecond=0)
        return ts_ms_from_datetime(nxt) - 1

    if interval == "1d":
        nxt = open_local + timedelta(days=1)
        nxt = nxt.replace(hour=0, minute=0, second=0, microsecond=0)
        return ts_ms_from_datetime(nxt) - 1

    seconds = _INTERVAL_TO_SECONDS.get(interval)
    if seconds is None:
        raise ValueError(f"Unsupported interval: {interval}")
    return bar_open_ts_ms + seconds * 1000 - 1


# =========================
# Shared headers (Event/Bar)
# =========================

class EventRecordHeader(BaseModel):
    """
    Common header for raw event records (snapshot, depth_diff, trade, ...).

    Timestamps:
    - event_ts_ms: exchange event time (if available; e.g., depthUpdate 'E')
    - tx_ts_ms: exchange transaction time (if available; e.g., depthUpdate 'T', aggTrade 'T')
    - recv_ts_ms: local receive time (always recommended)
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="event.v1")
    source: str = Field(default="binance_futures")

    symbol: str
    event_type: Literal["snapshot", "depth_diff", "trade", "kline"]

    event_ts_ms: int | None = None
    tx_ts_ms: int | None = None
    recv_ts_ms: int = Field(..., ge=0)


class BarFrameHeader(BaseModel):
    """
    Common header for any bar-aggregated frame (OHLCV, orderbook 1m features, trade 1m features, ...).

    Canonical join keys:
    - (symbol, bar_interval, bar_open_timestamp_ms)
    DataFrame index convention:
    - bar_open_datetime_tpe (Asia/Taipei)
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="barframe.v1")
    source: str = Field(default="binance_futures")

    symbol: str
    bar_interval: BarInterval

    bar_open_timestamp_ms: int = Field(..., ge=0)
    bar_close_timestamp_ms: int = Field(..., ge=0)

    bar_open_datetime_tpe: datetime
    bar_close_datetime_tpe: datetime

    @field_validator("bar_open_datetime_tpe", "bar_close_datetime_tpe")
    @classmethod
    def _ensure_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware (Asia/Taipei recommended)")
        return v


def make_bar_header(
    *,
    symbol: str,
    bar_interval: BarInterval,
    bar_open_timestamp_ms: int,
    source: str = "binance_futures",
    schema_version: str = "barframe.v1",
    tz: ZoneInfo = TZ_TPE,
) -> BarFrameHeader:
    """Build a BarFrameHeader from canonical open timestamp and interval."""
    open_dt = datetime_from_ts_ms(bar_open_timestamp_ms, tz=tz)
    close_ts = bar_close_ts_ms(bar_open_timestamp_ms, bar_interval, tz=tz)
    close_dt = datetime_from_ts_ms(close_ts, tz=tz)
    return BarFrameHeader(
        schema_version=schema_version,
        source=source,
        symbol=symbol,
        bar_interval=bar_interval,
        bar_open_timestamp_ms=bar_open_timestamp_ms,
        bar_close_timestamp_ms=close_ts,
        bar_open_datetime_tpe=open_dt,
        bar_close_datetime_tpe=close_dt,
    )


# =========================
# OHLCV bar schema (payload)
# =========================

class OHLCVRow(BarFrameHeader):
    """OHLCV bar row aligned to BarFrameHeader."""
    open: float
    high: float
    low: float
    close: float
    volume: float


# =================================
# Orderbook snapshot (raw + features)
# =================================

PriceQty = Tuple[float, float]


class SnapshotRecord(BaseModel):
    """
    Parsed orderbook snapshot (depth=N) fed into the feature engine.

    Notes:
    - bids should be sorted descending by price.
    - asks should be sorted ascending by price.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    header: EventRecordHeader

    depth: int = Field(..., ge=1)
    last_update_id: int | None = None

    bids: Sequence[PriceQty]
    asks: Sequence[PriceQty]

    @field_validator("bids")
    @classmethod
    def _validate_bids(cls, v: Sequence[PriceQty]) -> Sequence[PriceQty]:
        # Basic sanity only (keep light for performance).
        if len(v) == 0:
            raise ValueError("bids is empty")
        return v

    @field_validator("asks")
    @classmethod
    def _validate_asks(cls, v: Sequence[PriceQty]) -> Sequence[PriceQty]:
        if len(v) == 0:
            raise ValueError("asks is empty")
        return v


class OrderbookSnapshotFeatureRow(BarFrameHeader):
    """
    Orderbook snapshot-derived features aggregated to a bar interval (typically 1m).

    Keep payload intentionally flexible:
    - required: top-of-book state
    - optional: depth within bps, bps bins, slopes, quality flags, metadata
    """
    # Snapshot metadata (recommended)
    snapshot_depth: int = Field(..., ge=1)
    snapshot_last_update_id: int | None = None
    snapshot_event_ts_ms: int | None = None
    snapshot_recv_ts_ms: int = Field(..., ge=0)

    # Top-of-book (required)
    best_bid_p: float
    best_bid_q: float
    best_ask_p: float
    best_ask_q: float

    spread: float
    mid: float

    # Optional but common
    microprice: float | None = None
    imbalance_l1: float | None = None

    # Quality flags (recommended)
    flag_crossed_book: bool = False
    flag_bad_sorting: bool = False
    flag_depth_insufficient: bool = False
    flag_non_positive_qty: bool = False

    # Dynamic feature payload:
    # Store additional computed features (e.g., depth_bid_10bps, bin_0_5bps_bid_qty, ...)
    extra_features: dict[str, float | int] = Field(default_factory=dict)


# =========================
# Trade event schema (raw)
# =========================

class AggTradeRecord(BaseModel):
    """
    Binance futures aggTrade normalized record.

    Key semantics:
    - is_buyer_maker: if True, buyer was the maker (so trade was initiated by seller / aggressive sell).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    header: EventRecordHeader

    agg_trade_id: int
    price: float
    qty: float
    trade_ts_ms: int = Field(..., ge=0)

    is_buyer_maker: bool


# =========================
# Depth diff event schema (raw)
# =========================

class DepthDiffRecord(BaseModel):
    """
    Depth diff (depthUpdate) normalized record.

    Update id fields align with Binance depthUpdate semantics:
    - U: first update ID in event
    - u: final update ID in event
    - pu: previous final update ID
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    header: EventRecordHeader

    U: int
    u: int
    pu: int

    bids: Sequence[PriceQty]  # price, new_qty
    asks: Sequence[PriceQty]  # price, new_qty


__all__ = [
    "TZ_UTC",
    "TZ_TPE",
    "BarInterval",
    "EventRecordHeader",
    "BarFrameHeader",
    "make_bar_header",
    "floor_to_bar_open_in_tz",
    "bar_close_ts_ms",
    "OHLCVRow",
    "SnapshotRecord",
    "OrderbookSnapshotFeatureRow",
    "AggTradeRecord",
    "DepthDiffRecord",
]
