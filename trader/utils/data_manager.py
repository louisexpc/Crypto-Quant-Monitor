import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
import websockets
import aioredis
from collections import deque
from typing import Deque, Dict, List, Optional
from binance import AsyncClient, BinanceSocketManager
import logging
import pandas as pd

from crawler.listener import suppress_exceptions

from utils.utils import load_config
from utils.indicators import FeatureComputer, IndicatorLibrary
class DataManager:
    def __init__(
            self, 
            config: Dict,
            api_key: str,
            api_secret: str
        ):
        self.config = config
        self.api_key = api_key
        self.api_secret = api_secret
        self.market = (config.get('market')  or "spot").strip().lower()

        # Feature Precomputation
        self.feature_long_cfg = load_config(config.get('features', {}).get('long_feature_cfg_path', 'configs/feature_108_long.yaml'))
        self.feature_short_cfg = load_config(config.get('features', {}).get('short_feature_cfg_path', 'configs/feature_108_short.yaml'))
        self.indicator_lib = IndicatorLibrary()
        self.feature_computer_long = FeatureComputer(self.indicator_lib)
        self.feature_computer_short = FeatureComputer(self.indicator_lib)

        if self.market not in ("spot", "futures"):
            raise ValueError("market must be 'spot' or 'futures'")

        if ['symbols', 'interval', 'lookback'] not in config:
            raise ValueError("Config must include 'symbols', 'interval', and 'lookback'.")
        self.symbols = config['symbols']
        self.symbols = [s.strip().upper() for s in self.symbols] # 統一處理成大寫(Binance API要求)
        
        self.interval = config['interval']
        self.lookback = config['lookback']

        if self.interval not in ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]:
            raise ValueError(f"Unsupported interval {self.interval}. Supported intervals are: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d.")
        
        # Queues
        self.max_queue_size = 1000
        self.trade_queues: Dict[str, asyncio.Queue] = {symbol: asyncio.Queue(maxsize=self.max_queue_size) for symbol in self.symbols}
        self.trades_candle_queues: Dict[str, asyncio.Queue] = {symbol: asyncio.Queue(maxsize=self.max_queue_size) for symbol in self.symbols} # trades 聚合後的 K 線(根據 interval)
        self.kline_candle_queues: Dict[str, asyncio.Queue] = {symbol: asyncio.Queue(maxsize=self.max_queue_size) for symbol in self.symbols} # kline socket 的 K 線
        self.event_queues: Dict[str, asyncio.Queue] = {symbol: asyncio.Queue(maxsize=self.max_queue_size) for symbol in self.symbols}
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # 全域連線管理/冷卻
        self._reconnect_lock = asyncio.Lock()
        self._last_connect_ts = 0.0
        self._min_connect_interval = 2.0
        self._reconnect_history: Deque[float] = deque()
        self._cooldown_threshold = 30
        self._cooldown_seconds = 300.0

        # run time 
        self.queue_max: int = config.get('queue_max', 1000)
        self.health_timeout: float = config.get('health_timeout', 30.0)
        self.backoff_base: float = config.get('backoff_base', 1.0)
        self.backoff_max: float = config.get('backoff_max', 60.0)

        self._stop_event = asyncio.Event() # 用於停止所有任務: 初始化時狀態為未設定
        self._client: Optional[AsyncClient] = None
        self._bsm: Optional[BinanceSocketManager] = None # 可以共用多個 symbeol 的 socket manager(Klines/trades/orderbook)
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._writer_tasks: List[asyncio.Task] = []

        # Redis
        self.redis_url: str = config.get('redis_url', 'redis://localhost:6379/0')
        self._redis: Optional[aioredis.Redis] = None
    
    # ----------------------- helper functions -----------------------
    async def _acquire_connect_slot(self):
        """全域連線閘門：避免多條 WS 同時重連造成握手雪崩。"""
        async with self._reconnect_lock:
            now = time.time()
            wait = (self._last_connect_ts + self._min_connect_interval) - now
            if wait > 0:
                await asyncio.sleep(wait)
            # 加一點抖動
            await asyncio.sleep((asyncio.get_running_loop().time() % 0.3))
            self._last_connect_ts = time.time()
    def _record_reconnect_failure(self):
        now = time.time()
        self._reconnect_history.append(now)
        cutoff = now - 15 * 60
        while self._reconnect_history and self._reconnect_history[0] < cutoff:
            self._reconnect_history.popleft()

    async def _maybe_cooldown_on_flapping(self):
        if len(self._reconnect_history) >= self._cooldown_threshold:
            self.logger.warning("Too many reconnects in 15 min (%d). Cooling down %ss",
                            len(self._reconnect_history), int(self._cooldown_seconds))
            await asyncio.sleep(self._cooldown_seconds)
    def ts_to_datetime(self, ts: int) -> datetime:
        """Convert milliseconds timestamp to datetime object in UTC+8 (Asia/Taipei)."""
        return datetime.fromtimestamp(ts / 1000, tz=timezone(timedelta(hours=8)))
    def now_ts(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)
    def ts_to_min_floor(self, ts:int) -> int:
        """跟據 interval 將 timestamp 向下取整到分鐘/小時/天"""
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        if self.interval.endswith('m'):
            minutes = int(self.interval[:-1])
            floored_minute = (dt.minute // minutes) * minutes
            dt = dt.replace(minute=floored_minute, second=0, microsecond=0)
        elif self.interval.endswith('h'):
            hours = int(self.interval[:-1])
            floored_hour = (dt.hour // hours) * hours
            dt = dt.replace(hour=floored_hour, minute=0, second=0, microsecond=0)
        elif self.interval == '1d':
            dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(dt.timestamp() * 1000)
    # ----------------------- Kline Supervisor -----------------------
    async def kline_supervisor(self, symbol: str):
        backoff = self.backoff_base
        while not self._stop_event.is_set():
            try:
                await self._acquire_connect_slot()

                if self.market == "spot":
                    kline_ctx = self._bsm.kline_socket(symbol, interval=self.interval)
                elif self.market == "futures":
                    try:
                        from binance.enums import FuturesType
                    except ImportError:
                        raise ImportError("FuturesType enum not found in binance.enums. Please ensure you have the correct version of the Binance SDK installed.")
                    FUTURES_TYPE = FuturesType.USD_M
                    kline_ctx = self._bsm.kline_futures_socket(symbol, interval=self.interval, futures_type=FUTURES_TYPE)
                
                async with kline_ctx as stream:
                    self.logger.info("[kline:%s] kline_socket started (market=%s, interval=%s)", symbol, self.market, self.interval)
                    backoff = self.backoff_base  # Reset backoff on successful connection

                    while not self._stop_event.is_set():
                        try:
                            msg = await stream.recv()
                            ohlcv = {
                                'open_time': self.ts_to_datetime(msg['k']['t']),
                                'close_time': self.ts_to_datetime(msg['k']['T']),
                                'open': float(msg['k']['o']),
                                'high': float(msg['k']['h']),
                                'low': float(msg['k']['l']),
                                'close': float(msg['k']['c']),
                                'volume': float(msg['k']['v']),
                            }
                            self.logger.debug("[kline:%s] Received kline data: %s", symbol, ohlcv)
                            await self.kline_candle_queues[symbol].put((symbol, ohlcv))
                        except asyncio.TimeoutError:
                            self.logger.warning("[kline:%s] No kline data received for %.1f seconds, reconnecting...", symbol, self.health_timeout)
                            raise ConnectionError("Kline socket health check timeout")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("[kline:%s] supervisor error: %s", symbol, e)
                self._record_reconnect_failure()
                await self._maybe_cooldown_on_flapping()

            if self._stop_event.is_set():
                break
    # ----------------------- Trade Supervisor -----------------------
    async def trade_supervisor(self, symbol: str):
        backoff = self.backoff_base
        while not self._stop_event.is_set():
            try:
                await self._acquire_connect_slot()

                if self.market == "spot":
                    trade_ctx = self._bsm.trade_socket(symbol)
                elif self.market == "futures":
                    try:
                        from binance.enums import FuturesType
                    except ImportError:
                        raise ImportError("FuturesType enum not found in binance.enums. Please ensure you have the correct version of the Binance SDK installed.")
                    FUTURES_TYPE = FuturesType.USD_M
                    trade_ctx = self._bsm.aggtrade_futures_socket(symbol, futures_type=FUTURES_TYPE)
                
                async with trade_ctx as stream:
                    self.logger.info("[trade:%s] trade_socket started (market=%s)", symbol, self.market)
                    backoff = self.backoff_base  # Reset backoff on successful connection

                    while not self._stop_event.is_set():
                        try:
                            msg = await stream.recv()
                            await self.trade_queues[symbol].put(msg)
                        except asyncio.TimeoutError:
                            self.logger.warning("[trade:%s] No trade data received for %.1f seconds, reconnecting...", symbol, self.health_timeout)
                            raise ConnectionError("Trade socket health check timeout")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("[trade:%s] supervisor error: %s", symbol, e)
                self._record_reconnect_failure()
                await self._maybe_cooldown_on_flapping()

            if self._stop_event.is_set():
                break
    async def trade_aggregator(self, symbol: str):
        """
        聚合交易資料成 K 線
        1. 從 trade_queue 讀取交易資料
        2. 根據設定的 interval 聚合成 K 線
        3. 將聚合後的 K 線放入 candle_queue : ( "candle_closed", symbol, candle_dict )
        4. 持續進行直到停止事件被設定
        5. 重置狀態變數
        6. 結束任務
        """
        current_start = None
        candle = None
        try:
            while not self._stop_event.is_set():
                trade = await asyncio.wait_for(self.trade_queues[symbol].get(), timeout=1.0)

                trade_ts = trade['T']  # 交易時間戳 (毫秒)
                trade_price = float(trade['p'])
                trade_qty = float(trade['q'])

                # 根據需要的 interval timeframe 來聚合 trade 成 K 線
                bucket_start = self.ts_to_min_floor(trade_ts)
                if current_start is None:
                    # 初始化第一根 K 線
                    current_start = bucket_start
                    candle = {
                        'open_time': self.ts_to_datetime(current_start),
                        'close_time': None,
                        'open': trade_price,
                        'high': trade_price,
                        'low': trade_price,
                        'close': trade_price,
                        'volume': trade_qty,
                    }
                elif bucket_start != current_start:
                    # 完成當前 K 線，放入 candle_queue
                    candle['close_time'] = self.ts_to_datetime(current_start + self._interval_ms())
                    await self.trades_candle_queues[symbol].put(("candle_closed",symbol, candle))

                    # 開始新的 K 線
                    current_start = bucket_start
                    candle = {
                        'open_time': self.ts_to_datetime(current_start),
                        'close_time': None,
                        'open': trade_price,
                        'high': trade_price,
                        'low': trade_price,
                        'close': trade_price,
                        'volume': trade_qty,
                    }
                else:
                    # 更新當前 K 線
                    candle['high'] = max(candle['high'], trade_price)
                    candle['low'] = min(candle['low'], trade_price)
                    candle['close'] = trade_price
                    candle['volume'] += trade_qty
        except asyncio.CancelledError:
            pass
    # ----------------------- Redis Writer -----------------------
    async def klines_redis_writer(self):
        """
        從 kline_candle_queues 讀取 K 線資料，寫入 Redis
        支援 stream:candles:{symbol} 及 candles:{symbol}:{ts} 格式
        """
        while not self._stop_event.is_set():
            try:
                symbol, candle = await asyncio.wait_for(self.kline_candle_queues[symbol].get(), timeout=1.0)
            except asyncio.TimeoutError:
                self.logger.debug("No kline candle data to write to Redis, continuing...")
                continue
            stream_key = f"stream:candles:{symbol}"
            hash_key = f"candles:{symbol}:{int(candle['open_time'].timestamp() * 1000)}"
            # 寫入 Redis Stream
            await self._redis.xadd(stream_key, {
                'open_time': candle['open_time'].isoformat(),
                'close_time': candle['close_time'].isoformat(),
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume'],
            })
            # 寫入 Redis Hash
            await self._redis.hset(hash_key, mapping={
                'open_time': candle['open_time'].isoformat(),
                'close_time': candle['close_time'].isoformat(),
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume'],
            })
            self.logger.debug("Wrote kline candle to Redis for symbol %s at %s", symbol, candle['open_time'].isoformat())
            self.kline_candle_queues[symbol].task_done()
    async def trades_redis_writer(self):
        """
        從 trades_candle_queues 讀取 K 線資料，寫入 Redis
        支援 stream:candles:{symbol} 及 candles:{symbol}:{ts} 格式
        """
        while not self._stop_event.is_set():
            try:
                ev_type, symbol, candle = await asyncio.wait_for(self.trades_candle_queues[symbol].get(), timeout=1.0)
            except asyncio.TimeoutError:
                self.logger.debug("No trades candle data to write to Redis, continuing...")
                continue
            if ev_type != "candle_closed":
                self.logger.warning("Unknown event type %s in trades_redis_writer, skipping...", ev_type)
                continue
            stream_key = f"stream:candles:{symbol}"
            hash_key = f"candles:{symbol}:{int(candle['open_time'].timestamp() * 1000)}"
            # 寫入 Redis Stream
            await self._redis.xadd(stream_key, {
                'open_time': candle['open_time'].isoformat(),
                'close_time': candle['close_time'].isoformat(),
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume'],
            })
            # 寫入 Redis Hash
            await self._redis.hset(hash_key, mapping={
                'open_time': candle['open_time'].isoformat(),
                'close_time': candle['close_time'].isoformat(),
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume'],
            })
            self.logger.debug("Wrote trades candle to Redis for symbol %s at %s", symbol, candle['open_time'].isoformat())
            self.trades_candle_queues[symbol].task_done()
    # ----------------------- Feature Precomputation Functions -----------------------
    async def kline_feature_precomputation(self, symbol: str):
        """
        Design : 基於目前K 線資料量，採取 (feature computation + redis write) 架構，暫時不考慮 redis io stuck 影響
        """
        while not self._stop_event.is_set():
            try:
                symbol, candle = await asyncio.wait_for(self.kline_candle_queues[symbol].get(), timeout=1.0)
            except asyncio.TimeoutError:
                self.logger.debug("No kline candle data to write to Redis, continuing...")
                continue
            stream_key = f"stream:candles:{symbol}"
            hash_key = f"candles:{symbol}:{int(candle['open_time'].timestamp() * 1000)}"
            # 寫入 Redis Stream
            await self._redis.xadd(stream_key, {
                'open_time': candle['open_time'].isoformat(),
                'close_time': candle['close_time'].isoformat(),
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume'],
            })
            # 寫入 Redis Hash
            await self._redis.hset(hash_key, mapping={
                'open_time': candle['open_time'].isoformat(),
                'close_time': candle['close_time'].isoformat(),
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume'],
            })
            self.logger.debug("Wrote kline candle to Redis for symbol %s at %s", symbol, candle['open_time'].isoformat())

            # Feature computation
            # TODO: implement feature computation logic here
            df_raw = pd.DataFrame() # dump data, just for temporary placeholder
            long_plan = self.feature_long_cfg['features']['plan']
            short_plan = self.feature_short_cfg['features']['plan']
            self.indicator_lib._normalize_ohlcv(df_raw)
            long_feature = self.feature_computer_long.compute(long_plan, self.feature_long_cfg)
            short_feature = self.feature_computer_short.compute(short_plan, self.feature_short_cfg)
            
            self.kline_candle_queues[symbol].task_done()
        pass

    # ----------------------- Start  -----------------------
    async def start(self):
        self.logger.info("Starting AsyncClient (market=%s)...", self.market)
        self._client = await AsyncClient.create(api_key=self.api_key, api_secret=self.api_secret)
        # 嘗試帶 ws_params；舊版不支援則回退
        try:
            self._bsm = BinanceSocketManager(
                self._client,
                user_timeout=60,
                ws_params={
                    "ping_interval": 20,
                    "ping_timeout": 20,
                    "close_timeout": 10,
                    "open_timeout": 10,
                    "max_queue": None,
                },
            )
            self.logger.info("BinanceSocketManager created with ws_params keepalive & timeouts")
        except TypeError:
            self._bsm = BinanceSocketManager(self._client, user_timeout=60)
            self.logger.info("BinanceSocketManager created without ws_params (not supported in this version)")

        # Start supervisors
        for symbol in self.symbols:
            kline_task = asyncio.create_task(self.kline_supervisor(symbol))
            trade_task = asyncio.create_task(self.trade_supervisor(symbol))
            kline_redis_writer_task = asyncio.create_task(self.klines_redis_writer())
            trade_aggregator_task = asyncio.create_task(self.trade_aggregator(symbol))
            trades_redis_writer_task = asyncio.create_task(self.trades_redis_writer())
            self._tasks.extend([kline_task, trade_task, trades_redis_writer_task])
        self.logger.info("DataManager started for symbols: %s", self.symbols)

        # Start Redis connection
        try:
            self._redis = await aioredis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
            self.logger.info("Connected to Redis at %s", self.redis_url)
        except Exception as e:
            self.logger.error("Failed to connect to Redis at %s: %s", self.redis_url, e)
            raise

        self.logger.info("[service] started (market=%s)", self.market)
    # ----------------------- Stop  -----------------------
    async def stop(self):
        self._stop_event.set()
        all_tasks = []
        for t in self._tasks:
            t.cancel()
            all_tasks.append(t)
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        if self._client:
            with suppress_exceptions():
                await self._client.close_connection()
        self.logger.info("[service] stopped")