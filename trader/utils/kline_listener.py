# utils/kline_listener.py
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import AsyncIterator, Literal, Optional

import websockets  # pip install websockets


@dataclass(frozen=True)
class KlineClosedEvent:
    """A single kline-closed trigger event.

    bar_close_ts_ms uses Binance's `k.T` (close time in milliseconds).
    """

    symbol: str
    interval: str
    bar_close_ts_ms: int


class BinanceFuturesKlineListener:
    """Binance USDT-M Futures market stream listener (testnet or prod).

    Design goals
    ------------
    - Provide a *clean* trigger source for TradingBot (event-driven daemon).
    - Only emit events when the kline is closed (`k.x == true`).
    - Handle reconnects with exponential backoff + jitter.
    - Avoid local event queue blow-ups by dropping the oldest item when full.

    Important
    ---------
    This module is intentionally *stateless* regarding trading decisions.
    It does not fetch REST data, merge FNG, compute features, or place orders.
    """

    def __init__(
        self,
        symbol: str,
        interval: Literal['5m', '15m', '1h', '4h', '1d'] = "1h",
        testnet: bool = True,
        logger: Optional[logging.Logger] = None,
        queue_size: int = 1000,
        # Deduplicate identical close timestamps that may be re-delivered on reconnect.
        dedupe_same_ts: bool = True,
    ):
        self.symbol = symbol.upper()
        self.interval = interval
        self.testnet = testnet
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._stop_event = asyncio.Event()
        self._q: asyncio.Queue[KlineClosedEvent] = asyncio.Queue(maxsize=queue_size)

        base = "wss://fstream.binancefuture.com" if testnet else "wss://fstream.binance.com"
        stream = f"{self.symbol.lower()}@kline_{self.interval}"
        self._ws_url = f"{base}/ws/{stream}"

        self._task: Optional[asyncio.Task] = None

        self._dedupe_same_ts = dedupe_same_ts
        self._last_enqueued_ts_ms: Optional[int] = None
        self._last_msg_ts = 0.0  # for simple liveness diagnostics

    @property
    def ws_url(self) -> str:
        return self._ws_url

    @property
    def last_message_age_sec(self) -> float:
        """Seconds since last received WS message (diagnostic)."""
        if self._last_msg_ts <= 0:
            return float("inf")
        return max(0.0, time.time() - self._last_msg_ts)

    def start(self) -> None:
        """Start the background listener task."""
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run_loop(), name=f"kline_listener:{self.symbol}:{self.interval}")

    async def stop(self) -> None:
        """Stop the listener and drain resources."""
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def closed_kline_events(self) -> AsyncIterator[KlineClosedEvent]:
        """Async iterator of kline-closed events for TradingBot.

        Usage:
            async for evt in listener.closed_kline_events():
                ...
        """
        while not self._stop_event.is_set():
            evt = await self._q.get()
            yield evt

    async def _run_loop(self) -> None:
        backoff = 1.0
        backoff_max = 60.0

        while not self._stop_event.is_set():
            try:
                self.logger.info("WS connect: %s", self._ws_url)
                async with websockets.connect(self._ws_url, ping_interval=20, ping_timeout=20) as ws:
                    backoff = 1.0
                    async for msg in ws:
                        if self._stop_event.is_set():
                            break
                        self._last_msg_ts = time.time()
                        self._handle_message(msg)

            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("WS error; reconnecting...")

            # exponential backoff + jitter
            await asyncio.sleep(backoff + random.random() * 0.5)
            backoff = min(backoff * 2.0, backoff_max)

    def _handle_message(self, msg: str) -> None:
        try:
            data = json.loads(msg)
            k = data.get("k", {})
            if not k:
                return

            # Only trigger when the candle is finalized/closed.
            if not bool(k.get("x", False)):
                return

            bar_close_ts_ms = int(k["T"])  # close time in ms

            # Local dedupe: identical close timestamps can be re-delivered on reconnect.
            if self._dedupe_same_ts and self._last_enqueued_ts_ms == bar_close_ts_ms:
                return

            evt = KlineClosedEvent(
                symbol=self.symbol,
                interval=self.interval,
                bar_close_ts_ms=bar_close_ts_ms,
            )

            # If queue is full, drop the oldest to keep the latest triggers.
            if self._q.full():
                try:
                    _ = self._q.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            self._q.put_nowait(evt)
            self._last_enqueued_ts_ms = bar_close_ts_ms

        except Exception:
            self.logger.exception("Failed to parse WS message")
