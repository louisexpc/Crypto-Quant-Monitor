import asyncio
import json
import time
from typing import List, Dict, Optional, Any, Deque, Tuple
from datetime import datetime
import logging
from pathlib import Path
from collections import deque
import random

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

# ---------- monkey-patch ReconnectingWebsocket queue size (must run before creating sockets) ----------
# 目的：緩解 python-binance 內部 ReconnectingWebsocket 的 recv 佇列在尖峰時溢位
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
    return time.time()

def iso_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"

class suppress_exceptions:
    """用於 with 區塊內吞掉例外（僅限關閉時等非關鍵區段）"""
    def __enter__(self): pass
    def __exit__(self, exc_type, exc, tb): return True


# ---------- OrderBook in-memory 結構與邏輯 ----------
class OrderBook:
    """
    簡潔、可重建的本地 orderbook：
    - 使用 dict[str->float] 保存 price->qty（字串 key 避免浮點比較誤差；落檔時再轉型）
    - 依 Binance 官方規則套用 depth diff：
        每則事件包含 U(首序號), u(尾序號), pu(前一事件的 u)
        嚴格檢查序號連續性以避免 book 漂移
    """
    __slots__ = ("bids", "asks", "last_update_id")

    def __init__(self):
        self.bids: Dict[str, float] = {}  # 價→量
        self.asks: Dict[str, float] = {}
        self.last_update_id: Optional[int] = None

    @staticmethod
    def _apply_side(side: Dict[str, float], levels: List[List[str]]) -> None:
        """
        levels: [["price", "qty"], ...]
        qty == "0" 代表刪除該價位
        """
        for p, q in levels:
            if q == "0" or q == "0.00000000":
                side.pop(p, None)
            else:
                side[p] = float(q)

    def load_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        snapshot 需包含 bids/asks 與 lastUpdateId
        """
        self.bids.clear()
        self.asks.clear()
        for p, q in snapshot.get("bids", []):
            self.bids[p] = float(q)
        for p, q in snapshot.get("asks", []):
            self.asks[p] = float(q)
        self.last_update_id = int(snapshot["lastUpdateId"])

    def apply_diff_event(self, event: Dict[str, Any]) -> bool:
        """
        套用單筆 depthUpdate 事件（已完成「對齊」後才使用）。
        要求 pu == last_update_id，否則回 False 讓上層觸發重建。
        """
        data = event.get("data", event)
        if data.get("e") != "depthUpdate":
            return True  # 非 depth 事件，略過不視為錯誤

        U = int(data["U"])
        u = int(data["u"])
        pu = int(data.get("pu", U - 1))  # 若無 pu（理論上期貨都有），以 U-1 推估

        if self.last_update_id is None:
            return False

        # 線上模式：嚴格要求 pu == last_update_id
        if pu != self.last_update_id:
            return False

        # 應用事件
        bids = data.get("b", [])
        asks = data.get("a", [])
        self._apply_side(self.bids, bids)
        self._apply_side(self.asks, asks)
        self.last_update_id = u
        return True

    def topk(self, k: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        回傳 top-k asks 與 bids：
        - asks: 以價格由小到大
        - bids: 以價格由大到小
        """
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
    Production 等級的歷史爬蟲 Pipeline（單一 depth stream → 雙出口）：
      - 市場：
          market="spot"    使用 bsm.depth_socket / bsm.trade_socket
          market="futures" 使用 bsm.futures_depth_socket / bsm.aggtrade_futures_socket
      - 深度：
          以「先開 WS 緩衝 → 抓 REST 快照 → 對齊 U<=L+1<=u → 回放緩衝 → 在線對齊/套用」流程
          同一條 depth stream 同時：
             (A) 完整 diff 原文落檔
             (B) Online 維護本地 orderbook（OrderBook）
      - 交易：
          每筆 trades/aggTrades 完整落檔
      - 穩定性：
          全域連線閘門、退避 + 抖動、15 分鐘視窗冷卻
      - 檔案 I/O：
          以 thread pool 寫入 JSONL，避免阻塞事件迴圈
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
        self.symbols = [s.strip().upper() for s in symbols]  # 正規化 symbol（大寫）

        # queues: 分離 diff / trades 寫出
        self.diff_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
        self.trade_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)

        # 本地 orderbook 狀態：每個 symbol 一個 OrderBook
        self.books: Dict[str, OrderBook] = {s: OrderBook() for s in self.symbols}

        # writer 參數
        self.diff_batch_size = diff_batch_size
        self.diff_max_interval = diff_max_interval
        self.trade_batch_size = trade_batch_size
        self.trade_max_interval = trade_max_interval

        # snapshot
        self.snapshot_interval_sec = snapshot_interval_sec
        self.snapshot_top_k = snapshot_top_k

        # 檔案
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
        self._writer_tasks: List[asyncio.Task] = []

        # health/backoff
        self.health_timeout = health_timeout
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._last_msg_time: Dict[str, float] = {}

        # --- 全域連線節流 / 冷卻 ---
        self._reconnect_lock = asyncio.Lock()
        self._last_connect_ts = 0.0
        self._min_connect_interval = 2.0  # 任兩次開新 WS 的最小間隔秒數（全域）
        self._reconnect_history: Deque[float] = deque()
        self._cooldown_threshold = 30      # 15 分鐘內超過 N 次重連 → 進入冷卻
        self._cooldown_seconds = 300.0     # 冷卻 5 分鐘

        logging.info("Initialized AsyncBinanceStoragePipeline market=%s symbols=%s", self.market, self.symbols)

    # ---------------------------- 低階工具 ----------------------------

    async def _acquire_connect_slot(self):
        """
        全域連線閘門：確保每次「新開 WS」之間至少間隔 _min_connect_interval 秒；
        並加入極小抖動避免同時醒來造成握手尖峰。
        """
        async with self._reconnect_lock:
            now = time.time()
            wait = (self._last_connect_ts + self._min_connect_interval) - now
            if wait > 0:
                await asyncio.sleep(wait)
            # 非常小的抖動（用 loop 時鐘做簡單擾動）
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

    async def _write_lines(self, path: Path, lines: List[Dict[str, Any]]):
        """以 threadpool 非阻塞落檔（JSONL）"""
        def _sync_write(p: Path, batch: List[Dict[str, Any]]):
            with open(p, "a", encoding="utf-8") as f:
                for item in batch:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        await asyncio.to_thread(_sync_write, path, lines)

    # ---------------------------- REST 快照 ----------------------------

    async def _fetch_depth_snapshot(self, symbol: str, limit: int = 1000) -> Dict[str, Any]:
        """
        嘗試以 spot/futures 兩種方法取得 depth 快照（以相容 python-binance 1.0.30 為目標）。
        回傳格式需包含 bids/asks/lastUpdateId。
        """
        if self.market == "spot":
            return await self._client.get_order_book(symbol=symbol, limit=limit)

        # futures
        # 先嘗試現貨端點（部分安裝也可用）
        try:
            snap = await self._client.get_order_book(symbol=symbol, limit=limit)
            if "lastUpdateId" in snap:
                return snap
        except Exception:
            pass
        # 再用期貨端點
        snap = await self._client.futures_order_book(symbol=symbol, limit=limit)
        return snap

    # ---------------------------- Depth Supervisor ----------------------------

    async def _depth_supervised(self, symbol: str):
        """
        核心：單一 depth stream 雙出口（落 diff + 維護 book）
        啟動流程：
          1) 開 WS，開始緩衝 diff
          2) 抓 REST 快照，得到 L
          3) 丟掉所有 u <= L 的事件；若緩衝找到 U <= L+1 <= u，從此事件開始回放
          4) 若緩衝找不到 → 進入「在線對齊模式」：持續丟棄，直到遇到第一筆滿足 U <= L+1 <= u 才開始 apply
          5) 之後切到「線上模式」：要求 pu == last_update_id
        """
        backoff = self.backoff_base
        book = self.books[symbol]
        self._last_msg_time[symbol] = now_ts()

        while not self._stop_event.is_set():
            try:
                await self._acquire_connect_slot()

                # 選擇正確的 depth socket
                if self.market == "spot":
                    depth_ctx = self._bsm.depth_socket(symbol)
                else:
                    if not hasattr(self._bsm, "futures_depth_socket"):
                        raise RuntimeError("BinanceSocketManager 缺少 futures_depth_socket，請升級 python-binance")
                    if FUTURES_TYPE is not None:
                        depth_ctx = self._bsm.futures_depth_socket(symbol, depth="20", futures_type=FUTURES_TYPE)
                    else:
                        depth_ctx = self._bsm.futures_depth_socket(symbol, depth="20")

                buffer: Deque[Dict[str, Any]] = deque()
                snapshot_ready = False
                L: Optional[int] = None  # lastUpdateId from snapshot

                async with depth_ctx as stream:
                    logging.info("[depth:%s] depth_socket started (market=%s)", symbol, self.market)
                    backoff = self.backoff_base

                    # 啟動同時抓快照（與緩衝並行）
                    snapshot_task = asyncio.create_task(self._fetch_depth_snapshot(symbol, limit=1000))

                    # 等待 snapshot 完成並緩衝首批事件
                    while not snapshot_ready and not self._stop_event.is_set():
                        done, pending = await asyncio.wait(
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
                            # 收到 1 筆 WS 事件，先緩衝（bootstrap 階段也落檔，確保完整）
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

                    # 第二階段：對齊規則 U <= L+1 <= u，先嘗試用緩衝回放
                    while buffer and self._extract_u(buffer[0]) <= L:
                        buffer.popleft()

                    aligned = False
                    for _ in range(len(buffer)):
                        evt = buffer[0]
                        U, u = self._extract_Uu(evt)
                        if U <= L + 1 <= u:
                            # 找到第一筆可對齊事件：先 apply，last_update_id 變為 u
                            ok = book.apply_diff_event({
                                # 架構：apply_diff_event 需要完整事件；此處直接丟 evt 即可
                                **evt
                            })
                            if not ok:
                                # buffer 還是沒法對齊，跳出改用在線對齊
                                break
                            buffer.popleft()
                            aligned = True
                            logging.info("[depth:%s] aligned using bootstrap buffer U=%s u=%s L=%s", symbol, U, u, L)
                            break
                        else:
                            buffer.popleft()

                    # 回放剩餘緩衝（已經 aligned 的情況下才會走到這裡）
                    while aligned and buffer:
                        evt = buffer.popleft()
                        ok = book.apply_diff_event(evt)
                        if not ok:
                            # 若在回放中斷裂，強制重啟
                            raise RuntimeError("OrderBook sequence mismatch during buffer replay; restart required")

                    # 第三階段：正式在線
                    # 若尚未對齊 → 進入「在線對齊模式」
                    # 規則：
                    #   - 丟棄所有 u <= L 的事件
                    #   - 找到第一個滿足 U <= L+1 <= u 的事件，apply 後設 aligned=True
                    #   - 之後切換到正常 apply_diff_event（檢查 pu）
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

                        if not aligned:
                            data = msg.get("data", msg)
                            if data.get("e") != "depthUpdate":
                                continue  # 非 depth 事件丟掉（仍已落檔）

                            U = int(data["U"])
                            u = int(data["u"])

                            if u <= L:
                                # 仍是舊事件，丟掉
                                continue

                            if U <= L + 1 <= u:
                                # 找到第一筆可對齊事件：直接 apply，切換到線上模式
                                ok = book.apply_diff_event(msg)
                                if ok:
                                    aligned = True
                                    logging.info("[depth:%s] aligned using LIVE stream U=%s u=%s L=%s", symbol, U, u, L)
                                    continue
                                else:
                                    # 理論上不應該失敗；失敗就重啟
                                    raise RuntimeError("OrderBook alignment failed on LIVE event; restart required")
                            else:
                                # 還沒到可對齊的事件，繼續等
                                continue
                        else:
                            # 已對齊 → 走線上模式（嚴格 pu 檢查）
                            ok = book.apply_diff_event(msg)
                            if not ok:
                                logging.error("[depth:%s] sequence mismatch; forcing restart to re-snapshot/re-align", symbol)
                                break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.exception("[depth:%s] supervisor error: %s", symbol, e)
                self._record_reconnect_failure()
                await self._maybe_cooldown_on_flapping()

            if self._stop_event.is_set():
                break
            # 帶抖動的退避（避免同時醒來）
            jitter = random.uniform(0, 0.7 * backoff)
            wait = backoff + jitter
            logging.info("[depth:%s] restarting in %.2f s (base=%.2f, jitter=%.2f)", symbol, wait, backoff, jitter)
            await asyncio.sleep(wait)
            backoff = min(backoff * 2, self.backoff_max)

        logging.info("[depth:%s] supervisor exiting", symbol)

    async def _emit_depth_raw(self, symbol: str, msg: Dict[str, Any]):
        """將收到的 depth 事件完整落檔（JSONL），並加上 recv_ts/symbol/type 等外層欄位。"""
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

                # 選擇正確的 trade socket
                if self.market == "spot":
                    trade_ctx = self._bsm.trade_socket(symbol)  # 單筆成交（也可改 aggTrade）
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
            logging.info("[trade:%s] restarting in %.2f s (base=%.2f, jitter=%.2f)", symbol, wait, backoff, jitter)
            await asyncio.sleep(wait)
            backoff = min(backoff * 2, self.backoff_max)

        logging.info("[trade:%s] supervisor exiting", symbol)

    # ---------------------------- Writer Tasks ----------------------------

    async def _diff_writer(self):
        """
        專責寫 depth diff 原文（含 checkpoint），以批次減少 I/O 次數。
        """
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
                await self._write_lines(self.diff_log_path, batch)
                logging.debug("[diff_writer] flushed %d diffs (queue=%d)", len(batch), self.diff_queue.qsize())
        except asyncio.CancelledError:
            # 優雅收尾
            rem = []
            while True:
                try:
                    rem.append(self.diff_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if rem:
                await self._write_lines(self.diff_log_path, rem)
                logging.info("[diff_writer] flushed remaining %d diffs on cancel", len(rem))
            raise

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
                await self._write_lines(self.trades_path, batch)
                logging.debug("[trade_writer] flushed %d trades (queue=%d)", len(batch), self.trade_queue.qsize())
        except asyncio.CancelledError:
            rem = []
            while True:
                try:
                    rem.append(self.trade_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if rem:
                await self._write_lines(self.trades_path, rem)
                logging.info("[trade_writer] flushed remaining %d trades on cancel", len(rem))
            raise

    async def _snapshot_writer(self):
        """
        週期性輸出本地 orderbook 快照：
          - snapshot_{ts}.jsonl：保存該時刻每個 symbol 的全書/或 top-k
          - snapshot_latest.jsonl：覆蓋式最近快照
          - 並在 diff_log.jsonl 追加 checkpoint 記錄（能回放對齊）
        """
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(self.snapshot_interval_sec)
                ts = now_ts()
                to_write = []
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
                        "asks": asks,  # [(price, qty), ...]
                        "bids": bids
                    }
                    to_write.append(snap_out)

                    checkpoints.append({
                        "type": "snapshot_checkpoint",
                        "symbol": s,
                        "snapshot_file": None,  # 寫完才知道檔名，稍後覆值
                        "snapshot_ts": ts,
                        "lastUpdateId": book.last_update_id,
                        "written_ts": now_ts()
                    })

                if to_write:
                    snapshot_filename = str(self.snapshot_path_template).format(
                        ts=datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                    )
                    # 寫檔
                    await self._write_snapshot_file(snapshot_filename, to_write, append=True)
                    await self._write_snapshot_file(str(self.snapshot_latest_path), to_write, append=False)
                    logging.info("[snapshot_writer] wrote %d snapshots to %s", len(to_write), snapshot_filename)
                    # 在 diff_log.jsonl 追加 checkpoint（補上檔名）
                    for ck in checkpoints:
                        ck["snapshot_file"] = snapshot_filename
                    await self._write_lines(self.diff_log_path, checkpoints)
        except asyncio.CancelledError:
            raise

    async def _write_snapshot_file(self, filename: str, snapshots: List[Dict[str, Any]], append: bool = True):
        mode = "a" if append else "w"
        def _sync_write(path: str, snaps: List[Dict[str, Any]], mode_: str):
            with open(path, mode_, encoding="utf-8") as f:
                for s in snaps:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
        await asyncio.to_thread(_sync_write, filename, snapshots, mode)

    # ---------------------------- lifecycle ----------------------------

    async def start(self):
        logging.info("Starting AsyncClient (market=%s)...", self.market)
        self._client = await AsyncClient.create(api_key=self.api_key, api_secret=self.api_secret)

        # 儘量帶 websockets 參數；舊版不支援時回退
        try:
            self._bsm = BinanceSocketManager(
                self._client,
                user_timeout=60,
                ws_params={
                    "ping_interval": 20,
                    "ping_timeout": 20,
                    "close_timeout": 10,
                    "open_timeout": 10,
                    "max_queue": None,  # 交由我們自家節流；避免 websockets 內層爆
                },
            )
            logging.info("BinanceSocketManager created with ws_params keepalive & timeouts")
        except TypeError:
            self._bsm = BinanceSocketManager(self._client, user_timeout=60)
            logging.info("BinanceSocketManager created without ws_params (not supported in this version)")

        # 啟動各 symbol 的 depth 與 trade supervisors
        for s in self.symbols:
            self._last_msg_time[s] = now_ts()
            self._tasks.append(asyncio.create_task(self._depth_supervised(s)))
            self._tasks.append(asyncio.create_task(self._trade_supervised(s)))

        # writer tasks
        t_diff = asyncio.create_task(self._diff_writer())
        t_trade = asyncio.create_task(self._trade_writer())
        t_snapshot = asyncio.create_task(self._snapshot_writer())
        self._writer_tasks.extend([t_diff, t_trade, t_snapshot])
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
