# bsm_storage_pipeline.py
import asyncio
import json
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
from pathlib import Path
try:
    from binance import AsyncClient, BinanceSocketManager, DepthCacheManager
except Exception:
    try:
        from binance import AsyncClient, BinanceSocketManager
        from binance.depthcache import DepthCacheManager
    except Exception as e:
        raise ImportError("請安裝 python-binance 並確認版本支援 AsyncClient / DepthCacheManager") from e
# try:
#     from binance.ws.reconnecting_websocket import ReconnectingWebsocket
#     # 把這個值調大 (例如 10k) —— 根據你訂閱的 stream 數量與頻率決定
#     ReconnectingWebsocket.MAX_QUEUE_SIZE = 10_000
#     # Optional: 也能改 module-level constant if exists:
#     import binance.ws.reconnecting_websocket as _rw
#     if hasattr(_rw, "MAX_QUEUE_SIZE"):
#         _rw.MAX_QUEUE_SIZE = 10_000
#     print("Patched ReconnectingWebsocket.MAX_QUEUE_SIZE -> 10000")
# except Exception as e:
#     print("Cannot patch MAX_QUEUE_SIZE:", e)

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
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        # storage params
        diff_batch_size: int = 1000,
        diff_max_interval: float = 1.0,
        trade_batch_size: int = 500,
        trade_max_interval: float = 2.0,
        snapshot_interval_sec: int = 60,      # 每 N 秒寫一次 full snapshot (你要每分鐘 -> 60)
        snapshot_top_k: Optional[int] = None, # None => write full DepthCache.get_bids()/get_asks()
        # file paths
        diff_log_path: str = "diff_log.jsonl",
        snapshot_path_template: str = "snapshot_{ts}.jsonl",  # 會產生 timestamped snapshots；也會寫 snapshot_latest.jsonl
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
        self.diff_log_path = diff_log_path
        self.snapshot_path_template = snapshot_path_template
        self.snapshot_latest_path = snapshot_latest_path
        self.trades_path = trades_path

        # runtime
        self._client: Optional[AsyncClient] = None
        self._bsm: Optional[BinanceSocketManager] = None
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._supervisors: Dict[str, List[asyncio.Task]] = {}  # symbol -> [depthcache_supervisor, depth_diff_supervisor, trade_supervisor]
        self._writer_tasks: List[asyncio.Task] = []

        # health/backoff
        self.health_timeout = health_timeout
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._last_msg_time: Dict[str, float] = {}

    # ---------- DepthCacheManager supervisor (maintain local orderbook in-memory) ----------
    async def _depthcache_supervised(self, symbol: str):
        backoff = self.backoff_base
        while not self._stop_event.is_set():
            try:
                async with DepthCacheManager(self._client, symbol=symbol) as dcm:
                    print(f"[depthcache:{symbol}] DepthCacheManager started")
                    backoff = self.backoff_base
                    # dcm.recv() returns a DepthCache object (lib-specific)
                    while not self._stop_event.is_set():
                        try:
                            depth_cache = await dcm.recv()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            print(f"[depthcache:{symbol}] recv error: {e}")
                            break
                        else:
                            # update in-memory latest snapshot
                            st = now_ts()
                            self._last_msg_time[symbol] = st
                            # attempt to extract full bids/asks or use provided getters
                            try:
                                asks = depth_cache.get_asks()
                                bids = depth_cache.get_bids()
                                lastUpdateId = getattr(depth_cache, "update_id", None) or getattr(depth_cache, "lastUpdateId", None) or None
                            except Exception:
                                # fallback, try attributes
                                asks = getattr(depth_cache, "asks", None)
                                bids = getattr(depth_cache, "bids", None)
                                lastUpdateId = getattr(depth_cache, "update_time", None)
                            # store latest cache for snapshot writer
                            self.latest_depth_cache[symbol] = {
                                "symbol": symbol,
                                "ts": st,
                                "lastUpdateId": lastUpdateId,
                                "asks": asks,
                                "bids": bids
                            }
                            # total_received metric optional
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[depthcache:{symbol}] supervisor exception: {e}")
            # restart with backoff
            if self._stop_event.is_set():
                break
            print(f"[depthcache:{symbol}] restarting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.backoff_max)
        print(f"[depthcache:{symbol}] supervisor exiting")

    # ---------- Raw depth diff listener (subscribe to depth_socket and append diff events) ----------
    async def _depth_diff_supervised(self, symbol: str):
        backoff = self.backoff_base
        while not self._stop_event.is_set():
            try:
                async with self._bsm.depth_socket(symbol) as stream:
                    print(f"[depth_diff:{symbol}] depth_socket started")
                    backoff = self.backoff_base
                    while not self._stop_event.is_set():
                        try:
                            msg = await stream.recv()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            print(f"[depth_diff:{symbol}] recv exception: {e}")
                            break
                        else:
                            st = now_ts()
                            self._last_msg_time[symbol] = st
                            # push raw diff with metadata to diff_queue (backpressure if full)
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
                print(f"[depth_diff:{symbol}] supervisor error: {e}")
            if self._stop_event.is_set():
                break
            print(f"[depth_diff:{symbol}] restarting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.backoff_max)
        print(f"[depth_diff:{symbol}] supervisor exiting")

    # ---------- Trade supervisor (trade_socket) ----------
    async def _trade_supervised(self, symbol: str):
        backoff = self.backoff_base
        while not self._stop_event.is_set():
            try:
                async with self._bsm.trade_socket(symbol) as stream:
                    print(f"[trade:{symbol}] trade_socket started")
                    backoff = self.backoff_base
                    while not self._stop_event.is_set():
                        try:
                            msg = await stream.recv()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            print(f"[trade:{symbol}] recv exception: {e}")
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
                print(f"[trade:{symbol}] supervisor error: {e}")
            if self._stop_event.is_set():
                break
            print(f"[trade:{symbol}] restarting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.backoff_max)
        print(f"[trade:{symbol}] supervisor exiting")

    # ---------- Writers ----------
    async def _diff_writer(self):
        """
        Batch write diff_queue to diff_log_path (append-only). Also can accept snapshot checkpoints from snapshot writer.
        """
        try:
            while not self._stop_event.is_set():
                # wait for first element
                try:
                    first = await asyncio.wait_for(self.diff_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                batch = [first]
                start = time.time()
                # collect until batch size or timeout
                while len(batch) < self.diff_batch_size:
                    remaining = self.diff_max_interval - (time.time() - start)
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.diff_queue.get(), timeout=remaining)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                # write batch to file (synchronously in thread)
                await asyncio.to_thread(self._sync_write_diffs, batch)
                print(f"[diff_writer] flushed {len(batch)} diffs (queue={self.diff_queue.qsize()})")
        except asyncio.CancelledError:
            # flush remaining
            remaining = []
            while True:
                try:
                    remaining.append(self.diff_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if remaining:
                await asyncio.to_thread(self._sync_write_diffs, remaining)
                print(f"[diff_writer] flushed remaining {len(remaining)} diffs on cancel")
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
                print(f"[trade_writer] flushed {len(batch)} trades (queue={self.trade_queue.qsize()})")
        except asyncio.CancelledError:
            rem = []
            while True:
                try:
                    rem.append(self.trade_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if rem:
                await asyncio.to_thread(self._sync_write_trades, rem)
                print(f"[trade_writer] flushed remaining {len(rem)} trades on cancel")
            raise

    def _sync_write_trades(self, batch: List[Dict[str, Any]]):
        with open(self.trades_path, "a", encoding="utf-8") as f:
            for it in batch:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    async def _snapshot_writer(self):
        """
        每 snapshot_interval_sec 寫一次 full snapshot（使用 latest_depth_cache），
        並同時寫一個 checkpoint entry 到 diff_log，以便後續從 diff_log replay。
        """
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(self.snapshot_interval_sec)
                ts = now_ts()
                to_write = []
                for s in self.symbols:
                    snap = self.latest_depth_cache.get(s)
                    if not snap:
                        continue
                    # optionally trim top_k if set
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
                    # write snapshots (timestamped) AND update snapshot_latest
                    snapshot_filename = self.snapshot_path_template.format(ts=datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
                    await asyncio.to_thread(self._sync_write_snapshots, snapshot_filename, to_write)
                    await asyncio.to_thread(self._sync_write_snapshots, self.snapshot_latest_path, to_write, append=False)
                    print(f"[snapshot_writer] wrote {len(to_write)} snapshots to {snapshot_filename}")
                    # also write checkpoint into diff_log so diff replay knows snapshot boundary
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
                    # write checkpoint directly to diff_log to ensure ordering
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
        print("[service] started")

    async def stop(self):
        self._stop_event.set()
        # cancel supervisors
        all_tasks = []
        for tlist in self._supervisors.values():
            for t in tlist:
                t.cancel()
                all_tasks.append(t)
        # cancel writers
        for t in self._writer_tasks:
            t.cancel()
            all_tasks.append(t)
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        # close client
        if self._client:
            await self._client.close_connection()
        print("[service] stopped")


# # ---------- Usage example ----------
# async def main():
#     import dotenv
#     import os
#     dotenv.load_dotenv()  # load from .env if exists
    
#     api_key = os.getenv("BINANCE_API_KEY")
#     api_secret = os.getenv("BINANCE_API_SECRET")
#     symbols = ["btcusdt"]  # start small
#     svc = AsyncBinanceStoragePipeline(
#         api_key, api_secret, symbols,
#         diff_batch_size=1000, diff_max_interval=1.0,
#         trade_batch_size=500, trade_max_interval=2.0,
#         snapshot_interval_sec=60,
#         snapshot_top_k=None,  # None -> write full depth from DepthCache
#         diff_log_path="diff_log.jsonl",
#         snapshot_path_template="snapshot_{ts}.jsonl",
#         snapshot_latest_path="snapshot_latest.jsonl",
#         trades_path="trades.jsonl"
#     )
#     await svc.start()
#     try:
#         await asyncio.sleep(60 * 5)  # run for 5 minutes for demo
#     finally:
#         await svc.stop()

# if __name__ == "__main__":
#     asyncio.run(main())
