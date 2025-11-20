# pipeline_futures.py
import asyncio
import json
import time
from datetime import datetime, timezone
import websockets
import aioredis
from collections import deque

# ================ config =================
SYMBOL = "BTCUSDT"               # Binance Futures symbol
TRADE_WS = f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@trade"
TRADE_Q_SIZE = 20000
EVENT_Q_SIZE = 2048
REDIS_URL = "redis://localhost:6379/0"
# testing: use 1m candle; in prod change aggregation logic to 1h
CANDLE_DURATION_SECONDS = 60     # 1m for testing; set 3600 for 1h real run

# rolling windows (in number of candles)
ROLL_WINDOWS = [36, 72, 108]

# ================ helpers =================
def ts_to_min_floor(ts: float):
    # returns unix ts floored to minute (or floor of CANDLE_DURATION_SECONDS)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    floored_seconds = int(dt.timestamp() // CANDLE_DURATION_SECONDS * CANDLE_DURATION_SECONDS)
    return floored_seconds

def now_ts():
    return int(time.time())

# ================ workers =================
async def ws_trade_producer(trade_q: asyncio.Queue, stop_event: asyncio.Event):
    backoff = 1.0
    while not stop_event.is_set():
        try:
            async with websockets.connect(TRADE_WS, ping_interval=20, ping_timeout=10) as ws:
                print("WS connected to", TRADE_WS)
                backoff = 1.0
                async for raw in ws:
                    # raw is json string
                    data = json.loads(raw)
                    # Binance trade message fields: e (event), E (eventTime), p (price), q (qty), T (tradeTime)
                    trade = {
                        "price": float(data["p"]),
                        "qty": float(data["q"]),
                        "trade_ts": int(data["T"]) // 1000  # ms -> s
                    }
                    # backpressure: if queue full, this await blocks -> slows ingestion
                    await trade_q.put(trade)
                    if stop_event.is_set():
                        break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print("ws error:", e, "reconnect in", backoff)
            await asyncio.sleep(backoff + (backoff * 0.1))
            backoff = min(backoff * 2, 60.0)

async def kline_builder(trade_q: asyncio.Queue, event_q: asyncio.Queue, stop_event: asyncio.Event):
    """
    Build fixed-length candles based on CANDLE_DURATION_SECONDS.
    On candle close, put ("candle_closed", candle_dict) into event_q.
    """
    current_start = None
    candle = None

    while not stop_event.is_set():
        try:
            trade = await asyncio.wait_for(trade_q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        ts = trade["trade_ts"]
        bucket = ts_to_min_floor(ts)

        if current_start is None:
            current_start = bucket
            candle = {
                "start_ts": current_start,
                "open": trade["price"],
                "high": trade["price"],
                "low": trade["price"],
                "close": trade["price"],
                "volume": trade["qty"]
            }
        elif bucket != current_start:
            # candle closed
            closed = dict(candle)
            await event_q.put(("candle_closed", closed))
            # start new candle with this trade
            current_start = bucket
            candle = {
                "start_ts": current_start,
                "open": trade["price"],
                "high": trade["price"],
                "low": trade["price"],
                "close": trade["price"],
                "volume": trade["qty"]
            }
        else:
            p = trade["price"]
            candle["high"] = max(candle["high"], p)
            candle["low"] = min(candle["low"], p)
            candle["close"] = p
            candle["volume"] += trade["qty"]

        trade_q.task_done()

async def redis_writer(event_q: asyncio.Queue, redis, stop_event: asyncio.Event):
    """
    On candle_closed event, write to:
      - Redis Stream: stream:candles:{symbol}
      - Redis Hash: candles:{symbol}:{ts}
    """
    stream_key = f"stream:candles:{SYMBOL}"
    while not stop_event.is_set():
        try:
            ev_type, payload = await asyncio.wait_for(event_q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        if ev_type == "candle_closed":
            c = payload
            ts = c["start_ts"]
            hkey = f"candles:{SYMBOL}:{ts}"
            # write hash (stringify numbers)
            await redis.hset(hkey, mapping={
                "start_ts": str(c["start_ts"]),
                "open": str(c["open"]),
                "high": str(c["high"]),
                "low": str(c["low"]),
                "close": str(c["close"]),
                "volume": str(c["volume"])
            })
            # set retention (example: keep 10000 keys)
            await redis.expire(hkey, 60*60*24*7)  # one week

            # append to stream (durable)
            await redis.xadd(stream_key, fields={
                "start_ts": str(c["start_ts"]),
                "open": str(c["open"]),
                "high": str(c["high"]),
                "low": str(c["low"]),
                "close": str(c["close"]),
                "volume": str(c["volume"])
            })

            print(f"[redis_writer] wrote candle {ts}")
        event_q.task_done()

async def precompute_worker(event_q: asyncio.Queue, redis, stop_event: asyncio.Event):
    """
    For each closed candle, compute rolling features (simple example: rolling average close)
    Maintain in-memory deque per symbol for efficiency (but persist features to Redis)
    """
    # We'll maintain a deque of last max(ROLL_WINDOWS) closes
    max_window = max(ROLL_WINDOWS)
    closes = deque(maxlen=max_window)

    while not stop_event.is_set():
        try:
            ev_type, payload = await asyncio.wait_for(event_q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            await asyncio.sleep(0.1)
            continue

        if ev_type == "candle_closed":
            c = payload
            close = float(c["close"])
            ts = c["start_ts"]
            closes.append(close)

            features = {}
            # compute rolling mean for each window if enough data, else None
            for w in ROLL_WINDOWS:
                if len(closes) >= w:
                    arr = list(closes)[-w:]
                    rolling_mean = sum(arr) / w
                else:
                    rolling_mean = None
                features[f"rolling_{w}"] = "" if rolling_mean is None else str(rolling_mean)

            # persist features to Redis hash
            fkey = f"features:{SYMBOL}:{ts}"
            await redis.hset(fkey, mapping=features)
            await redis.expire(fkey, 60*60*24*7)
            print(f"[precompute] ts={ts} features={features}")

        event_q.task_done()

# ================ main & graceful shutdown =================
async def main():
    trade_q = asyncio.Queue(maxsize=TRADE_Q_SIZE)
    event_q = asyncio.Queue(maxsize=EVENT_Q_SIZE)
    stop_event = asyncio.Event()

    redis = await aioredis.from_url(REDIS_URL)

    tasks = [
        asyncio.create_task(ws_trade_producer(trade_q, stop_event)),
        asyncio.create_task(kline_builder(trade_q, event_q, stop_event)),
        asyncio.create_task(redis_writer(event_q, redis, stop_event)),
        asyncio.create_task(precompute_worker(event_q, redis, stop_event)),
    ]

    # signal handlers
    loop = asyncio.get_running_loop()
    for s in (asyncio.constants.SIGINT, asyncio.constants.SIGTERM):
        loop.add_signal_handler(s, lambda: stop_event.set())

    try:
        await stop_event.wait()
        print("Stop requested, waiting for queues to drain (timeout 10s)...")
        await asyncio.wait_for(trade_q.join(), timeout=10.0)
        await asyncio.wait_for(event_q.join(), timeout=10.0)
    except asyncio.TimeoutError:
        print("Drain timeout, cancelling tasks.")
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await redis.close()

if __name__ == "__main__":
    asyncio.run(main())
