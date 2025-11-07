import asyncio
import json
import time
from typing import List, Dict, Optional, Any, Deque, Tuple
from datetime import datetime, date, timedelta
import logging
from pathlib import Path
from collections import deque
import random
from zoneinfo import ZoneInfo

# --------- config: timezone ----------
# 所有「以日期為單位」的行為（例如每日壓縮、每日檔名）都以 Asia/Taipei 為準
LOCAL_TZ = ZoneInfo("Asia/Taipei")

# --------- basic logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- try imports ----------
try:
    from binance import AsyncClient, BinanceSocketManager
except Exception as e:
    raise ImportError("請安裝 python-binance 並確認版本支援 AsyncClient / BinanceSocketManager") from e

# Optional import of FuturesType enum
FUTURES_TYPE = None
try:
    from binance.enums import FuturesType
    FUTURES_TYPE = FuturesType.USD_M
except Exception:
    logging.info("binance.enums.FuturesType not available; futures socket calls 將省略 futures_type 參數")

# ---------- monkey-patch ReconnectingWebsocket queue size ----------
# 緩解 python-binance 內部 ReconnectingWebsocket 的 recv 佇列在尖峰時溢位
try:
    from binance.ws.reconnecting_websocket import ReconnectingWebsocket
    ReconnectingWebsocket.MAX_QUEUE_SIZE = 2000
    import binance.ws.reconnecting_websocket as _rw
    if hasattr(_rw, "MAX_QUEUE_SIZE"):
        _rw.MAX_QUEUE_SIZE = 2000
    logging.info("Patched ReconnectingWebsocket.MAX_QUEUE_SIZE -> 2000")
except Exception as e:
    logging.warning("Cannot patch MAX_QUEUE_SIZE: %s", e)


# ---------- Helper ----------
def now_ts() -> float:
    # 儲存在 JSON 內的 epoch 秒數，保持 UTC 基準（方便回放與跨時區計算）
    return time.time()


def iso_ts_local() -> str:
    # 若要人類可讀時間，一律輸出 Asia/Taipei ISO 格式
    return datetime.now(LOCAL_TZ).isoformat()


class suppress_exceptions:
    """用於 with 區塊內吞掉例外（僅限關閉時等非關鍵區段）"""
    def __enter__(self): pass
    def __exit__(self, exc_type, exc, tb): return True


# ---------- OrderBook in-memory 結構 ----------
class OrderBook:
    """
    簡潔、可重建的本地 orderbook。

    - 使用 dict[str->float] 保存 price->qty（字串 key 避免浮點誤差）
    - 依 Binance 規則套用 depth diff：
        每則事件包含 U(首序), u(尾序), pu(前一事件的 u)
        嚴格檢查序號連續性以避免 book 漂移
    """
    __slots__ = ("bids", "asks", "last_update_id")

    def __init__(self):
        self.bids: Dict[str, float] = {}
        self.asks: Dict[str, float] = {}
        self.last_update_id: Optional[int] = None

    @staticmethod
    def _apply_side(side: Dict[str, float], levels: List[List[str]]) -> None:
        """levels: [[price, qty], ...]; qty == "0" 表示刪除該價位"""
        for p, q in levels:
            if q == "0" or q == "0.00000000":
                side.pop(p, None)
            else:
                side[p] = float(q)

    def load_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """以 REST snapshot 初始化 orderbook，要求包含 bids/asks/lastUpdateId。"""
        self.bids.clear()
        self.asks.clear()
        for p, q in snapshot.get("bids", []):
            self.bids[p] = float(q)
        for p, q in snapshot.get("asks", []):
            self.asks[p] = float(q)
        self.last_update_id = int(snapshot["lastUpdateId"])

    def apply_diff_event(self, event: Dict[str, Any]) -> bool:
        """
        套用單筆 depthUpdate（已完成對齊之後）。

        嚴格要求 pu == last_update_id，否則回傳 False 讓上層觸發重建。
        """
        data = event.get("data", event)
        if data.get("e") != "depthUpdate":
            return True  # 非 depth 事件不視為錯誤

        U = int(data["U"])
        u = int(data["u"])
        pu = int(data.get("pu", U - 1))

        if self.last_update_id is None:
            return False

        if pu != self.last_update_id:
            return False

        bids = data.get("b", [])
        asks = data.get("a", [])
        self._apply_side(self.bids, bids)
        self._apply_side(self.asks, asks)
        self.last_update_id = u
        return True

    def topk(self, k: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """回傳 top-k asks/bids；asks 由小到大、bids 由大到小。"""
        if k <= 0:
            return ([], [])
        asks_sorted = sorted(((float(p), q) for p, q in self.asks.items()), key=lambda x: x[0])[:k]
        bids_sorted = sorted(((float(p), q) for p, q in self.bids.items()), key=lambda x: x[0], reverse=True)[:k]
        return asks_sorted, bids_sorted

    def full_book(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        asks_sorted = sorted(((float(p), q) for p, q in self.asks.items()), key=lambda x: x[0])
        bids_sorted = sorted(((float(p), q) for p, q in self.bids.items()), key=lambda x: x[0], reverse=True)
        return asks_sorted, bids_sorted


# ---------- Main class ----------
class AsyncBinanceStoragePipeline:
    """
    Production 等級歷史爬蟲 Pipeline（維持 main.py 介面）：

    - 單一 depth stream：
        * 完整落檔各筆 depth diff（JSONL）
        * 同步維護 in-memory orderbook（預設用 1000 層 snapshot 足夠回測）
    - 單一 trade stream：
        * 完整落檔每筆成交
    - 檔案策略：
        * diff:    diff_log_YYYYMMDD.jsonl
        * trades:  trades_YYYYMMDD.jsonl
        * snapshot: snapshot_{YYYYMMDDThhmmss}.jsonl + snapshot_latest.jsonl
        * 每日壓縮：archive_YYYYMMDD.zip 包含前一天所有上述檔案
          （Asia/Taipei 為日界）
    - 穩定性：
        * 全域連線節流、退避 + 抖動、15 分鐘 flapping 冷卻
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
        # file paths (作為「基底檔名」，實際會加上 YYYYMMDD)
        diff_log_path: str = "diff_log.jsonl",
        snapshot_path_template: str = "snapshot_{ts}.jsonl",
        snapshot_latest_path: str = "snapshot_latest.jsonl",
        trades_path: str = "trades.jsonl",
        # runtime params
        queue_max: int = 200000,
        health_timeout: float = 30.0,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        # archive params
        delete_after_archive: bool = False,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.market = (market or "spot").strip().lower()
        if self.market not in ("spot", "futures"):
            raise ValueError("market must be 'spot' or 'futures'")
        self.symbols = [s.strip().upper() for s in symbols]

        # queues
        self.diff_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
        self.trade_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)

        # orderbooks
        self.books: Dict[str, OrderBook] = {s: OrderBook() for s in self.symbols}

        # writer params
        self.diff_batch_size = diff_batch_size
        self.diff_max_interval = diff_max_interval
        self.trade_batch_size = trade_batch_size
        self.trade_max_interval = trade_max_interval

        # snapshot
        self.snapshot_interval_sec = snapshot_interval_sec
        self.snapshot_top_k = snapshot_top_k

        # files
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.diff_log_base = self.out_dir / diff_log_path
        self.snapshot_path_template = self.out_dir / snapshot_path_template
        self.snapshot_latest_path = self.out_dir / snapshot_latest_path
        self.trades_log_base = self.out_dir / trades_path

        # runtime
        self._client: Optional[AsyncClient] = None
        self._bsm: Optional[BinanceSocketManager] = None
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._writer_tasks: List[asyncio.Task] = []

        # health/backoff
        self.health_timeout = health_timeout
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._last_msg_time: Dict[str, float] = {}

        # 全域連線節流 / 冷卻
        self._reconnect_lock = asyncio.Lock()
        self._last_connect_ts = 0.0
        self._min_connect_interval = 2.0
        self._reconnect_history: Deque[float] = deque()
        self._cooldown_threshold = 30
        self._cooldown_seconds = 300.0

        # 每日壓縮
        self.delete_after_archive = delete_after_archive
        self._archive_last_day: Optional[date] = None

        logging.info("Initialized AsyncBinanceStoragePipeline market=%s symbols=%s", self.market, self.symbols)

    # ---------------------------- 小工具 ----------------------------

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
            logging.warning("Too many reconnects in 15 min (%d). Cooling down %ss",
                            len(self._reconnect_history), int(self._cooldown_seconds))
            await asyncio.sleep(self._cooldown_seconds)

    def _daily_path(self, base: Path, day: date) -> Path:
        """將基底檔名轉成當日檔案：xxx_YYYYMMDD.ext"""
        stem = base.stem
        suffix = base.suffix or ".jsonl"
        name = f"{stem}_{day.strftime('%Y%m%d')}{suffix}"
        return base.with_name(name)

    async def _write_lines(self, path: Path, lines: List[Dict[str, Any]]):
        """以 threadpool 非阻塞落檔（JSONL）。"""
        def _sync_write(p: Path, batch: List[Dict[str, Any]]):
            with open(p, "a", encoding="utf-8") as f:
                for item in batch:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        await asyncio.to_thread(_sync_write, path, lines)

    # ---------------------------- REST 快照 ----------------------------

    async def _fetch_depth_snapshot(self, symbol: str, limit: int = 1000) -> Dict[str, Any]:
        """抓 depth snapshot（spot/futures 相容），需包含 bids/asks/lastUpdateId。"""
        if self.market == "spot":
            return await self._client.get_order_book(symbol=symbol, limit=limit)

        # futures: 先嘗試現貨接口，失敗再 futures_order_book
        try:
            snap = await self._client.get_order_book(symbol=symbol, limit=limit)
            if "lastUpdateId" in snap:
                return snap
        except Exception:
            pass
        snap = await self._client.futures_order_book(symbol=symbol, limit=limit)
        return snap

    # ---------------------------- Depth Supervisor ----------------------------

    async def _depth_supervised(self, symbol: str):
        """
        單一 depth stream：
        - 啟動：WS + snapshot 並行，依官方建議對齊 U<=L+1<=u
        - 線上模式：嚴格檢查 pu == last_update_id
        - 每筆事件推入 diff_queue，由 writer 寫 daily 檔
        """
        backoff = self.backoff_base
        book = self.books[symbol]
        self._last_msg_time[symbol] = now_ts()

        while not self._stop_event.is_set():
            try:
                await self._acquire_connect_slot()

                # 選擇 depth socket
                if self.market == "spot":
                    depth_ctx = self._bsm.depth_socket(symbol)
                else:
                    if not hasattr(self._bsm, "futures_depth_socket"):
                        raise RuntimeError("BinanceSocketManager 缺少 futures_depth_socket，請升級 python-binance")
                    if FUTURES_TYPE is not None:
                        depth_ctx = self._bsm.futures_depth_socket(symbol, futures_type=FUTURES_TYPE)
                    else:
                        depth_ctx = self._bsm.futures_depth_socket(symbol)

                buffer: Deque[Dict[str, Any]] = deque()
                snapshot_ready = False
                L: Optional[int] = None  # lastUpdateId from snapshot

                async with depth_ctx as stream:
                    logging.info("[depth:%s] depth_socket started (market=%s)", symbol, self.market)
                    backoff = self.backoff_base

                    snapshot_task = asyncio.create_task(self._fetch_depth_snapshot(symbol, limit=1000))

                    # bootstrap：snapshot + 緩衝
                    while not snapshot_ready and not self._stop_event.is_set():
                        done, _ = await asyncio.wait(
                            {snapshot_task, asyncio.create_task(stream.recv())},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if snapshot_task in done:
                            try:
                                snap = snapshot_task.result()
                                if "lastUpdateId" not in snap:
                                    raise RuntimeError("Depth snapshot missing lastUpdateId")
                                L = int(snap["lastUpdateId"])
                                book.load_snapshot(snap)
                                snapshot_ready = True
                                logging.info("[depth:%s] snapshot loaded lastUpdateId=%s", symbol, L)
                            except Exception as e:
                                logging.exception("[depth:%s] snapshot error: %s", symbol, e)
                                raise
                        else:
                            # 收到 1 筆 WS 事件，先緩衝並落檔
                            try:
                                msg = (list(done)[0]).result()
                            except Exception as e:
                                logging.exception("[depth:%s] recv during bootstrap: %s", symbol, e)
                                raise
                            self._last_msg_time[symbol] = now_ts()
                            await self._emit_depth_raw(symbol, msg)
                            buffer.append(msg)

                    if not snapshot_ready or L is None:
                        raise RuntimeError("Snapshot not ready but stream active; restarting...")

                    # 用緩衝對齊：丟掉 u <= L
                    while buffer and self._extract_u(buffer[0]) <= L:
                        buffer.popleft()

                    aligned = False
                    # 找第一筆 U <= L+1 <= u
                    for _ in range(len(buffer)):
                        evt = buffer[0]
                        U, u = self._extract_Uu(evt)
                        if U <= L + 1 <= u:
                            ok = book.apply_diff_event(evt)
                            buffer.popleft()
                            if ok:
                                aligned = True
                                logging.info("[depth:%s] aligned using bootstrap buffer U=%s u=%s L=%s",
                                             symbol, U, u, L)
                            break
                        else:
                            buffer.popleft()

                    # 回放剩餘緩衝（在已對齊情況下）
                    while aligned and buffer:
                        evt = buffer.popleft()
                        ok = book.apply_diff_event(evt)
                        if not ok:
                            raise RuntimeError("OrderBook sequence mismatch during buffer replay; restart required")

                    # 線上 loop
                    while not self._stop_event.is_set():
                        try:
                            msg = await stream.recv()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logging.exception("[depth:%s] recv exception: %s", symbol, e)
                            break

                        self._last_msg_time[symbol] = now_ts()
                        await self._emit_depth_raw(symbol, msg)

                        data = msg.get("data", msg)
                        if data.get("e") != "depthUpdate":
                            continue

                        if not aligned:
                            # 在線對齊模式：直到遇到 U <= L+1 <= u
                            U = int(data["U"])
                            u = int(data["u"])
                            if u <= L:
                                continue
                            if U <= L + 1 <= u:
                                ok = book.apply_diff_event(msg)
                                if ok:
                                    aligned = True
                                    logging.info("[depth:%s] aligned using LIVE stream U=%s u=%s L=%s",
                                                 symbol, U, u, L)
                                    continue
                                else:
                                    raise RuntimeError("OrderBook alignment failed on LIVE event; restart required")
                            else:
                                continue
                        else:
                            # 正常線上模式
                            ok = book.apply_diff_event(msg)
                            if not ok:
                                logging.error("[depth:%s] sequence mismatch; forcing restart", symbol)
                                break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.exception("[depth:%s] supervisor error: %s", symbol, e)
                self._record_reconnect_failure()
                await self._maybe_cooldown_on_flapping()

            if self._stop_event.is_set():
                break
            jitter = random.uniform(0, 0.7 * backoff)
            wait = backoff + jitter
            logging.info("[depth:%s] restarting in %.2f s (base=%.2f, jitter=%.2f)",
                         symbol, wait, backoff, jitter)
            await asyncio.sleep(wait)
            backoff = min(backoff * 2, self.backoff_max)

        logging.info("[depth:%s] supervisor exiting", symbol)

    async def _emit_depth_raw(self, symbol: str, msg: Dict[str, Any]):
        """推送 depth diff 到 queue，由 writer 按日期寫檔。"""
        wrapped = {
            "type": "depth_diff",
            "symbol": symbol,
            "recv_ts": now_ts(),
            "msg": msg,
        }
        try:
            await self.diff_queue.put(wrapped)
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _extract_Uu(event: Dict[str, Any]) -> Tuple[int, int]:
        data = event.get("data", event)
        return int(data["U"]), int(data["u"])

    @staticmethod
    def _extract_u(event: Dict[str, Any]) -> int:
        data = event.get("data", event)
        return int(data["u"])

    # ---------------------------- Trade Supervisor ----------------------------

    async def _trade_supervised(self, symbol: str):
        backoff = self.backoff_base
        while not self._stop_event.is_set():
            try:
                await self._acquire_connect_slot()

                # 選擇正確 trade socket
                if self.market == "spot":
                    trade_ctx = self._bsm.trade_socket(symbol)
                else:
                    trade_ctx = None
                    if hasattr(self._bsm, "aggtrade_futures_socket"):
                        if FUTURES_TYPE is not None:
                            trade_ctx = self._bsm.aggtrade_futures_socket(symbol, futures_type=FUTURES_TYPE)
                        else:
                            trade_ctx = self._bsm.aggtrade_futures_socket(symbol)
                    elif hasattr(self._bsm, "futures_aggtrade_socket"):
                        if FUTURES_TYPE is not None:
                            trade_ctx = self._bsm.futures_aggtrade_socket(symbol, futures_type=FUTURES_TYPE)
                        else:
                            trade_ctx = self._bsm.futures_aggtrade_socket(symbol)
                    else:
                        raise RuntimeError("BinanceSocketManager 缺少 aggtrade_futures_socket/futures_aggtrade_socket，請升級 python-binance")

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
                            payload = {
                                "type": "trade",
                                "symbol": symbol,
                                "recv_ts": now_ts(),
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
                self._record_reconnect_failure()
                await self._maybe_cooldown_on_flapping()

            if self._stop_event.is_set():
                break
            jitter = random.uniform(0, 0.7 * backoff)
            wait = backoff + jitter
            logging.info("[trade:%s] restarting in %.2f s (base=%.2f, jitter=%.2f)",
                         symbol, wait, backoff, jitter)
            await asyncio.sleep(wait)
            backoff = min(backoff * 2, self.backoff_max)

        logging.info("[trade:%s] supervisor exiting", symbol)

    # ---------------------------- Writer Tasks ----------------------------

    async def _diff_writer(self):
        """批次寫 depth diff，按 recv_ts 的 Asia/Taipei 日期切 daily 檔。"""
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

                ts = batch[0]["recv_ts"]
                day = datetime.fromtimestamp(ts, LOCAL_TZ).date()
                path = self._daily_path(self.diff_log_base, day)
                await self._write_lines(path, batch)
                logging.debug("[diff_writer] flushed %d diffs to %s (queue=%d)",
                              len(batch), path.name, self.diff_queue.qsize())
        except asyncio.CancelledError:
            rem = []
            while True:
                try:
                    rem.append(self.diff_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if rem:
                ts = rem[0]["recv_ts"]
                day = datetime.fromtimestamp(ts, LOCAL_TZ).date()
                path = self._daily_path(self.diff_log_base, day)
                await self._write_lines(path, rem)
                logging.info("[diff_writer] flushed remaining %d diffs on cancel to %s",
                             len(rem), path.name)
            raise

    async def _trade_writer(self):
        """批次寫 trades，按 recv_ts 的 Asia/Taipei 日期切 daily 檔。"""
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

                ts = batch[0]["recv_ts"]
                day = datetime.fromtimestamp(ts, LOCAL_TZ).date()
                path = self._daily_path(self.trades_log_base, day)
                await self._write_lines(path, batch)
                logging.debug("[trade_writer] flushed %d trades to %s (queue=%d)",
                              len(batch), path.name, self.trade_queue.qsize())
        except asyncio.CancelledError:
            rem = []
            while True:
                try:
                    rem.append(self.trade_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if rem:
                ts = rem[0]["recv_ts"]
                day = datetime.fromtimestamp(ts, LOCAL_TZ).date()
                path = self._daily_path(self.trades_log_base, day)
                await self._write_lines(path, rem)
                logging.info("[trade_writer] flushed remaining %d trades on cancel to %s",
                             len(rem), path.name)
            raise

    async def _snapshot_writer(self):
        """
        週期性輸出本地 orderbook 快照與 checkpoint：

        - snapshot_{YYYYMMDDThhmmss}.jsonl（檔名用 Asia/Taipei 時間）
        - snapshot_latest.jsonl（覆蓋）
        - 在當日 diff_log_YYYYMMDD.jsonl 追加 snapshot_checkpoint 記錄
        """
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(self.snapshot_interval_sec)
                ts = now_ts()
                now_local = datetime.fromtimestamp(ts, LOCAL_TZ)
                day = now_local.date()

                snapshots = []
                checkpoints = []
                for s, book in self.books.items():
                    if book.last_update_id is None:
                        continue
                    if self.snapshot_top_k:
                        asks, bids = book.topk(self.snapshot_top_k)
                    else:
                        asks, bids = book.full_book()
                    snap_out = {
                        "symbol": s,
                        "snapshot_ts": ts,
                        "lastUpdateId": book.last_update_id,
                        "asks": asks,
                        "bids": bids,
                    }
                    snapshots.append(snap_out)
                    checkpoints.append({
                        "type": "snapshot_checkpoint",
                        "symbol": s,
                        "snapshot_file": None,  # 稍後填入
                        "snapshot_ts": ts,
                        "lastUpdateId": book.last_update_id,
                        "written_ts": now_ts(),
                    })

                if not snapshots:
                    continue

                # 以本地時間產出檔名
                ts_str = now_local.strftime("%Y%m%dT%H%M%S")
                snapshot_filename = str(self.snapshot_path_template).format(ts=ts_str)
                snapshot_path = Path(snapshot_filename)

                await self._write_snapshot_file(snapshot_path, snapshots, append=True)
                await self._write_snapshot_file(self.snapshot_latest_path, snapshots, append=False)

                logging.info("[snapshot_writer] wrote %d snapshots to %s",
                             len(snapshots), snapshot_path.name)

                # checkpoint 寫入當日 diff_log_YYYYMMDD.jsonl
                for ck in checkpoints:
                    ck["snapshot_file"] = snapshot_path.name
                diff_daily_path = self._daily_path(self.diff_log_base, day)
                await self._write_lines(diff_daily_path, checkpoints)
        except asyncio.CancelledError:
            raise

    async def _write_snapshot_file(self, path: Path, snapshots: List[Dict[str, Any]], append: bool = True):
        mode = "a" if append else "w"

        def _sync_write(p: Path, snaps: List[Dict[str, Any]], mode_: str):
            with open(p, mode_, encoding="utf-8") as f:
                for s in snaps:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

        await asyncio.to_thread(_sync_write, str(path), snapshots, mode)

    # ---------------------------- Daily Archiver ----------------------------

    async def _daily_archiver(self):
        """
        每天打包「前一天」的所有檔案成 zip（Asia/Taipei 為日界）：

        - diff_log_YYYYMMDD.jsonl
        - trades_YYYYMMDD.jsonl
        - snapshot_YYYYMMDD*.jsonl
        => archive_YYYYMMDD.zip

        之後呼叫 _upload_zip_to_cloud(zip_path) 讓你上傳到雲端。
        若 delete_after_archive=True，成功後刪除原始 JSONL。
        """
        if self._archive_last_day is None:
            self._archive_last_day = (datetime.now(LOCAL_TZ).date() - timedelta(days=1))

        while not self._stop_event.is_set():
            try:
                today = datetime.now(LOCAL_TZ).date()
                target_day = today - timedelta(days=1)

                while self._archive_last_day is not None and self._archive_last_day <= target_day:
                    day = self._archive_last_day
                    await self._archive_one_day(day)
                    self._archive_last_day = day + timedelta(days=1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.exception("[archiver] error: %s", e)

            # 每 5 分鐘檢查一次
            await asyncio.sleep(300)

        logging.info("[archiver] exiting")

    async def _archive_one_day(self, day: date):
        files: List[Path] = []

        diff_path = self._daily_path(self.diff_log_base, day)
        if diff_path.exists():
            files.append(diff_path)

        trade_path = self._daily_path(self.trades_log_base, day)
        if trade_path.exists():
            files.append(trade_path)

        day_prefix = day.strftime("%Y%m%d")
        for p in self.out_dir.glob(f"snapshot_{day_prefix}*.jsonl"):
            files.append(p)

        if not files:
            logging.info("[archiver] no files to archive for %s", day)
            return

        zip_name = f"archive_{day.strftime('%Y%m%d')}.zip"
        zip_path = self.out_dir / zip_name
        if zip_path.exists():
            logging.info("[archiver] archive already exists for %s: %s", day, zip_path)
            return

        import zipfile

        def _make_zip():
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    zf.write(f, arcname=f.name)

        await asyncio.to_thread(_make_zip)
        logging.info("[archiver] created %s with %d files", zip_path.name, len(files))

        await self._upload_zip_to_cloud(zip_path)

        if self.delete_after_archive:
            for f in files:
                try:
                    f.unlink()
                except Exception as e:
                    logging.warning("[archiver] failed to remove %s: %s", f, e)

    async def _upload_zip_to_cloud(self, zip_path: Path):
        """
        上傳 zip 至雲端的 hook（預設只 log，不做實際上傳）。

        你可以在自己的程式中 monkey-patch:
            pipeline._upload_zip_to_cloud = my_async_uploader

        例如整合:
            - AWS S3 (aioboto3 / boto3 包 async wrapper)
            - GCP Storage
            - Azure Blob
        """
        try:
            size = zip_path.stat().st_size
        except Exception:
            size = -1
        logging.info("[archiver] zip ready for upload: %s (size=%d bytes)", zip_path, size)

    # ---------------------------- lifecycle ----------------------------

    async def start(self):
        logging.info("Starting AsyncClient (market=%s)...", self.market)
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
            logging.info("BinanceSocketManager created with ws_params keepalive & timeouts")
        except TypeError:
            self._bsm = BinanceSocketManager(self._client, user_timeout=60)
            logging.info("BinanceSocketManager created without ws_params (not supported in this version)")

        # 啟動 depth / trade supervisors
        for s in self.symbols:
            self._last_msg_time[s] = now_ts()
            self._tasks.append(asyncio.create_task(self._depth_supervised(s)))
            self._tasks.append(asyncio.create_task(self._trade_supervised(s)))

        # writer & archiver tasks
        t_diff = asyncio.create_task(self._diff_writer())
        t_trade = asyncio.create_task(self._trade_writer())
        t_snapshot = asyncio.create_task(self._snapshot_writer())
        t_archive = asyncio.create_task(self._daily_archiver())

        self._writer_tasks.extend([t_diff, t_trade, t_snapshot, t_archive])
        self._tasks.extend(self._writer_tasks)

        logging.info("[service] started (market=%s)", self.market)

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
        logging.info("[service] stopped")
