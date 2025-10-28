# bsm_storage_pipeline.py
import asyncio
import json
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
from pathlib import Path

# --------- basic logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- try imports ----------
try:
    from binance import AsyncClient, BinanceSocketManager, DepthCacheManager
except Exception:
    try:
        from binance import AsyncClient, BinanceSocketManager
        from binance.depthcache import DepthCacheManager
    except Exception as e:
        raise ImportError("請安裝 python-binance 並確認版本支援 AsyncClient / DepthCacheManager") from e

# Optional import of FuturesType enum
FUTURES_TYPE = None
try:
    from binance.enums import FuturesType
    FUTURES_TYPE = FuturesType.USD_M
except Exception:
    # FUTURES_TYPE stays None if enum not available in this installation
    logging.info("binance.enums.FuturesType not available; futures socket calls will omit futures_type arg")

# ---------- monkey-patch ReconnectingWebsocket queue size (must run before creating sockets) ----------
try:
    from binance.ws.reconnecting_websocket import ReconnectingWebsocket
    ReconnectingWebsocket.MAX_QUEUE_SIZE = 1000  # adjust as you need
    import binance.ws.reconnecting_websocket as _rw
    if hasattr(_rw, "MAX_QUEUE_SIZE"):
        _rw.MAX_QUEUE_SIZE = 1000
    logging.info("Patched ReconnectingWebsocket.MAX_QUEUE_SIZE -> 1000")
except Exception as e:
    logging.warning("Cannot patch MAX_QUEUE_SIZE: %s", e)

# ---------- try import futures depth cache manager (optional) ----------
_FuturesDepthCacheManager = None
try:
    # try likely locations
    try:
        from binance.ws.depthcache import FuturesDepthCacheManager
        _FuturesDepthCacheManager = FuturesDepthCacheManager
    except Exception:
        try:
            from binance.depthcache import FuturesDepthCacheManager
            _FuturesDepthCacheManager = FuturesDepthCacheManager
        except Exception:
            _FuturesDepthCacheManager = None
except Exception:
    _FuturesDepthCacheManager = None

# ---------- Helper ----------
def now_ts():
    return time.time()

def iso_ts():
    return datetime.utcnow().isoformat() + "Z"

class suppress_exceptions:
    def __enter__(self): pass
    def __exit__(self, exc_type, exc, tb): return True

# ---------- Main class ----------
class AsyncBinanceStoragePipeline:
    """
    Storage pipeline that can operate on two markets:
      - market="spot"    -> uses DepthCacheManager, bsm.depth_socket, bsm.trade_socket
      - market="futures" -> uses FuturesDepthCacheManager (if available), bsm.futures_depth_socket, aggtrade_futures_socket
    Other pipeline behavior (diff/trade queues, snapshot writer) unchanged.
    """
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        market: str = "spot",  # "spot" or "futures"
        # storage params
        out_dir: str = "output",
        diff_batch_size: int = 1000,
        diff_max_interval: float = 1.0,
        trade_batch_size: int = 500,
        trade_max_interval: float = 2.0,
        snapshot_interval_sec: int = 60,
        snapshot_top_k: Optional[int] = None,
        # file paths
        diff_log_path: str = "diff_log.jsonl",
        snapshot_path_template: str = "snapshot_{ts}.jsonl",
        snapshot_latest_path: str = "snapshot_latest.jsonl",
        trades_path: str = "trades.jsonl",
        # runtime params
        queue_max: int = 200000,
        health_timeout: float = 30.0,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.market = (market or "spot").strip().lower()
        if self.market not in ("spot", "futures"):
            raise ValueError("market must be 'spot' or 'futures'")
        # normalize symbols to Binance format (uppercase)
        self.symbols = [s.strip().upper() for s in symbols]

        # queues: separate pipelines
        self.diff_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
        self.trade_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
        # snapshot uses in-memory latest cache (per symbol)
        self.latest_depth_cache: Dict[str, Any] = {}

        # batch/write params
        self.diff_batch_size = diff_batch_size
        self.diff_max_interval = diff_max_interval
        self.trade_batch_size = trade_batch_size
        self.trade_max_interval = trade_max_interval
        self.snapshot_interval_sec = snapshot_interval_sec
        self.snapshot_top_k = snapshot_top_k

        # files
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.diff_log_path = self.out_dir / diff_log_path
        self.snapshot_path_template = self.out_dir / snapshot_path_template
        self.snapshot_latest_path = self.out_dir / snapshot_latest_path
        self.trades_path = self.out_dir / trades_path

        # runtime
        self._client: Optional[AsyncClient] = None
        self._bsm: Optional[BinanceSocketManager] = None
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._supervisors: Dict[str, List[asyncio.Task]] = {}
        self._writer_tasks: List[asyncio.Task] = []

        # health/backoff
        self.health_timeout = health_timeout
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._last_msg_time: Dict[str, float] = {}

        # select depth cache manager class for market
        if self.market == "spot":
            self._DepthCacheManagerClass = DepthCacheManager
        else:
            if _FuturesDepthCacheManager is None:
                raise ImportError("FuturesDepthCacheManager not available in python-binance installation. "
                                  "Please upgrade python-binance or use market='spot'.")
            self._DepthCacheManagerClass = _FuturesDepthCacheManager

        logging.info("Initialized AsyncBinanceStoragePipeline market=%s symbols=%s", self.market, self.symbols)

    # ---------- DepthCacheManager supervisor (maintain local orderbook in-memory) ----------
    async def _depthcache_supervised(self, symbol: str):
        backoff = self.backoff_base
        while not self._stop_event.is_set():
            try:
                # use selected DepthCacheManager class
                async with self._DepthCacheManagerClass(self._client, symbol=symbol) as dcm:
                    logging.info("[depthcache:%s] DepthCacheManager started (market=%s)", symbol, self.market)
                    backoff = self.backoff_base
                    while not self._stop_event.is_set():
                        try:
                            depth_cache = await dcm.recv()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logging.exception("[depthcache:%s] recv error: %s", symbol, e)
                            break
                        else:
                            st = now_ts()
                            self._last_msg_time[symbol] = st
                            try:
                                asks = depth_cache.get_asks()
                                bids = depth_cache.get_bids()
                                lastUpdateId = getattr(depth_cache, "update_id", None) or getattr(depth_cache, "lastUpdateId", None) or None
                            except Exception:
                                asks = getattr(depth_cache, "asks", None)
                                bids = getattr(depth_cache, "bids", None)
                                lastUpdateId = getattr(depth_cache, "update_time", None)
                            self.latest_depth_cache[symbol] = {
                                "symbol": symbol,
                                "ts": st,
                                "lastUpdateId": lastUpdateId,
                                "asks": asks,
                                "bids": bids
                            }
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.exception("[depthcache:%s] supervisor exception: %s", symbol, e)
            if self._stop_event.is_set():
                break
            logging.info("[depthcache:%s] restarting in %s s", symbol, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.backoff_max)
        logging.info("[depthcache:%s] supervisor exiting", symbol)

    # ---------- Raw depth diff listener (subscribe to depth_socket and append diff events) ----------
    async def _depth_diff_supervised(self, symbol: str):
        backoff = self.backoff_base
        while not self._stop_event.is_set():
            try:
                # choose correct socket method based on market
                if self.market == "spot":
                    depth_ctx = self._bsm.depth_socket(symbol)
                else:
                    # futures depth socket method (with optional futures_type)
                    if not hasattr(self._bsm, "futures_depth_socket"):
                        raise RuntimeError("BinanceSocketManager does not expose futures_depth_socket; upgrade python-binance")
                    if FUTURES_TYPE is not None:
                        depth_ctx = self._bsm.futures_depth_socket(symbol, depth="20", futures_type=FUTURES_TYPE)
                    else:
                        depth_ctx = self._bsm.futures_depth_socket(symbol,depth="20")

                async with depth_ctx as stream:
                    logging.info("[depth_diff:%s] depth_socket started (market=%s)", symbol, self.market)
                    backoff = self.backoff_base
                    while not self._stop_event.is_set():
                        try:
                            msg = await stream.recv()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logging.exception("[depth_diff:%s] recv exception: %s", symbol, e)
                            break
                        else:
                            st = now_ts()
                            self._last_msg_time[symbol] = st
                            payload = {
                                "type": "depth_diff",
                                "symbol": symbol,
                                "recv_ts": st,
                                "msg": msg
                            }
                            try:
                                await self.diff_queue.put(payload)
                            except asyncio.CancelledError:
                                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.exception("[depth_diff:%s] supervisor error: %s", symbol, e)
            if self._stop_event.is_set():
                break
            logging.info("[depth_diff:%s] restarting in %s s", symbol, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.backoff_max)
        logging.info("[depth_diff:%s] supervisor exiting", symbol)

    # ---------- Trade supervisor (trade_socket) ----------
    async def _trade_supervised(self, symbol: str):
        backoff = self.backoff_base
        while not self._stop_event.is_set():
            try:
                # choose correct trade socket
                if self.market == "spot":
                    trade_ctx = self._bsm.trade_socket(symbol)
                else:
                    # prefer aggtrade_futures_socket for per-symbol futures trades
                    trade_ctx = None
                    if hasattr(self._bsm, "aggtrade_futures_socket"):
                        if FUTURES_TYPE is not None:
                            trade_ctx = self._bsm.aggtrade_futures_socket(symbol, futures_type=FUTURES_TYPE)
                        else:
                            trade_ctx = self._bsm.aggtrade_futures_socket(symbol)
                    elif hasattr(self._bsm, "futures_aggtrade_socket"):
                        # alternative method name in some versions
                        if FUTURES_TYPE is not None:
                            trade_ctx = self._bsm.futures_aggtrade_socket(symbol, futures_type=FUTURES_TYPE)
                        else:
                            trade_ctx = self._bsm.futures_aggtrade_socket(symbol)
                    else:
                        raise RuntimeError("BinanceSocketManager does not expose aggtrade_futures_socket/futures_aggtrade_socket; upgrade python-binance")

                async with trade_ctx as stream:
                    logging.info("[trade:%s] trade_socket started (market=%s)", symbol, self.market)
                    backoff = self.backoff_base
                    while not self._stop_event.is_set():
                        try:
                            msg = await stream.recv()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logging.exception("[trade:%s] recv exception: %s", symbol, e)
                            break
                        else:
                            st = now_ts()
                            self._last_msg_time[symbol] = st
                            payload = {
                                "type": "trade",
                                "symbol": symbol,
                                "recv_ts": st,
                                "msg": msg
                            }
                            try:
                                await self.trade_queue.put(payload)
                            except asyncio.CancelledError:
                                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.exception("[trade:%s] supervisor error: %s", symbol, e)
            if self._stop_event.is_set():
                break
            logging.info("[trade:%s] restarting in %s s", symbol, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.backoff_max)
        logging.info("[trade:%s] supervisor exiting", symbol)

    # ---------- Writers (unchanged) ----------
    async def _diff_writer(self):
        try:
            while not self._stop_event.is_set():
                try:
                    first = await asyncio.wait_for(self.diff_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                batch = [first]
                start = time.time()
                while len(batch) < self.diff_batch_size:
                    remaining = self.diff_max_interval - (time.time() - start)
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.diff_queue.get(), timeout=remaining)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                await asyncio.to_thread(self._sync_write_diffs, batch)
                logging.info("[diff_writer] flushed %d diffs (queue=%d)", len(batch), self.diff_queue.qsize())
        except asyncio.CancelledError:
            remaining = []
            while True:
                try:
                    remaining.append(self.diff_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if remaining:
                await asyncio.to_thread(self._sync_write_diffs, remaining)
                logging.info("[diff_writer] flushed remaining %d diffs on cancel", len(remaining))
            raise

    def _sync_write_diffs(self, batch: List[Dict[str, Any]]):
        with open(self.diff_log_path, "a", encoding="utf-8") as f:
            for item in batch:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    async def _trade_writer(self):
        try:
            while not self._stop_event.is_set():
                try:
                    first = await asyncio.wait_for(self.trade_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                batch = [first]
                start = time.time()
                while len(batch) < self.trade_batch_size:
                    remaining = self.trade_max_interval - (time.time() - start)
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.trade_queue.get(), timeout=remaining)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                await asyncio.to_thread(self._sync_write_trades, batch)
                logging.info("[trade_writer] flushed %d trades (queue=%d)", len(batch), self.trade_queue.qsize())
        except asyncio.CancelledError:
            rem = []
            while True:
                try:
                    rem.append(self.trade_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if rem:
                await asyncio.to_thread(self._sync_write_trades, rem)
                logging.info("[trade_writer] flushed remaining %d trades on cancel", len(rem))
            raise

    def _sync_write_trades(self, batch: List[Dict[str, Any]]):
        with open(self.trades_path, "a", encoding="utf-8") as f:
            for it in batch:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    async def _snapshot_writer(self):
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(self.snapshot_interval_sec)
                ts = now_ts()
                to_write = []
                for s in self.symbols:
                    snap = self.latest_depth_cache.get(s)
                    if not snap:
                        continue
                    if self.snapshot_top_k:
                        snap_out = {
                            "symbol": s,
                            "snapshot_ts": ts,
                            "lastUpdateId": snap.get("lastUpdateId"),
                            "asks": snap["asks"][: self.snapshot_top_k] if snap["asks"] else None,
                            "bids": snap["bids"][: self.snapshot_top_k] if snap["bids"] else None,
                        }
                    else:
                        snap_out = {
                            "symbol": s,
                            "snapshot_ts": ts,
                            "lastUpdateId": snap.get("lastUpdateId"),
                            "asks": snap.get("asks"),
                            "bids": snap.get("bids")
                        }
                    to_write.append(snap_out)
                if to_write:
                    snapshot_filename = str(self.snapshot_path_template).format(ts=datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
                    await asyncio.to_thread(self._sync_write_snapshots, snapshot_filename, to_write)
                    await asyncio.to_thread(self._sync_write_snapshots, str(self.snapshot_latest_path), to_write, append=False)
                    logging.info("[snapshot_writer] wrote %d snapshots to %s", len(to_write), snapshot_filename)
                    checkpoint_entries = []
                    for snap in to_write:
                        checkpoint_entries.append({
                            "type": "snapshot_checkpoint",
                            "symbol": snap["symbol"],
                            "snapshot_file": snapshot_filename,
                            "snapshot_ts": snap["snapshot_ts"],
                            "lastUpdateId": snap["lastUpdateId"],
                            "written_ts": now_ts()
                        })
                    await asyncio.to_thread(self._sync_write_diffs, checkpoint_entries)
        except asyncio.CancelledError:
            raise

    def _sync_write_snapshots(self, filename: str, snapshots: List[Dict[str, Any]], append: bool = True):
        mode = "a" if append else "w"
        with open(filename, mode, encoding="utf-8") as f:
            for s in snapshots:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ---------- lifecycle ----------
    async def start(self):
        # create AsyncClient and BinanceSocketManager
        logging.info("Starting AsyncClient (market=%s)...", self.market)
        self._client = await AsyncClient.create(api_key=self.api_key, api_secret=self.api_secret)
        self._bsm = BinanceSocketManager(self._client, user_timeout=60)

        # start supervisors per symbol
        for s in self.symbols:
            self._last_msg_time[s] = now_ts()
            tasks = []
            tasks.append(asyncio.create_task(self._depthcache_supervised(s)))
            tasks.append(asyncio.create_task(self._depth_diff_supervised(s)))
            tasks.append(asyncio.create_task(self._trade_supervised(s)))
            self._supervisors[s] = tasks
            self._tasks.extend(tasks)

        # start writer tasks
        t_diff = asyncio.create_task(self._diff_writer())
        t_trade = asyncio.create_task(self._trade_writer())
        t_snapshot = asyncio.create_task(self._snapshot_writer())
        self._writer_tasks.extend([t_diff, t_trade, t_snapshot])
        self._tasks.extend(self._writer_tasks)
        logging.info("[service] started (market=%s)", self.market)

    async def stop(self):
        self._stop_event.set()
        all_tasks = []
        for tlist in self._supervisors.values():
            for t in tlist:
                t.cancel()
                all_tasks.append(t)
        for t in self._writer_tasks:
            t.cancel()
            all_tasks.append(t)
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        if self._client:
            await self._client.close_connection()
        logging.info("[service] stopped")
