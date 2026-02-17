import asyncio
import logging
import os
import signal
import time
from pathlib import Path
from typing import Optional, List, Literal

import dotenv
from pydantic import BaseModel
import yaml
import argparse
import pandas as pd
# Module imports (do not modify other modules)
from utils.data_collector import ExchangeDataCollector, ExchangeConfig
from utils.trader import Trader
from utils.kline_listener import BinanceFuturesKlineListener, KlineClosedEvent
from utils.state_manager import BotStateStore
from indicators.feature_computer import FeatureComputer
from predictor.predictor import Predictor
from strategy.strategy import SNRLiveStrategy, SNRCfg, Candle
from utils.discord_bot import DiscordNotifier
dotenv.load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent

class APIKeyConfig(BaseModel):
    api_key: str
    api_secret: str
    discord_bot_token: str
    discord_channel_id: int

class ProcessGuard:
    """A minimal single-instance guard using a pidfile.

    Why
    ---
    In production, running two daemons pointing at the same account can cause:
    - duplicated triggers,
    - duplicated orders,
    - state corruption.

    This guard prevents starting a second instance when one is already running.
    """

    def __init__(self, pidfile: str):
        self.pidfile = Path(pidfile)

    def acquire(self) -> None:
        """Create pidfile or raise RuntimeError if another live pid is detected."""
        self.pidfile.parent.mkdir(parents=True, exist_ok=True)

        if self.pidfile.exists():
            try:
                old_pid = int(self.pidfile.read_text(encoding="utf-8").strip())
            except Exception:
                old_pid = None

            if old_pid and self._pid_is_running(old_pid):
                raise RuntimeError(f"Another daemon is already running (pid={old_pid}, pidfile={self.pidfile}).")

        # Write our pid (best effort).
        self.pidfile.write_text(str(os.getpid()), encoding="utf-8")

    def release(self) -> None:
        """Remove pidfile (best effort)."""
        try:
            if self.pidfile.exists():
                self.pidfile.unlink()
        except Exception:
            pass

    @staticmethod
    def _pid_is_running(pid: int) -> bool:
        """Return True if process exists."""
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _parse_interval_to_ms(interval: str) -> int:
    """Parse interval strings like '1h', '15m', '1d' into milliseconds.

    Supported units:
      - ms, s, m, h, d
    """
    s = interval.strip().lower()
    if not s:
        raise ValueError("Empty interval")

    # Allow forms like '60m' / '1h' / '3600s'
    num = ""
    unit = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            unit += ch

    if not num or not unit:
        raise ValueError(f"Invalid interval format: {interval!r}")

    n = int(num)
    unit = unit.strip()

    mult = {
        "ms": 1,
        "s": 1000,
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
    }.get(unit)

    if mult is None:
        raise ValueError(f"Unsupported interval unit: {unit!r} (use ms/s/m/h/d)")

    return n * mult


def _extract_latest_ts_ms(df) -> Optional[int]:
    """Best-effort extraction of 'latest timestamp' from a dataframe.

    This helper is used ONLY for REST-finalize retry (to wait until REST data
    has caught up after a WS close event). It does not affect idempotency keys.

        Priority:

            1) df['kline_close_timestamp_ms'] column
            2) df['timestamp'] column (seconds or ms; best-effort)
            3) df.index (usually datetime in Asia/Taipei tz)
    """
    if df is None or len(df) == 0:
        return None

    try:
        if "kline_close_timestamp_ms" in df.columns:
            return int(df["kline_close_timestamp_ms"].iloc[-1])

        if "timestamp" in df.columns:
            x = int(df["timestamp"].iloc[-1])
            # heuristic: treat 13-digit as ms, 10-digit as seconds
            return x if x > 10_000_000_000 else x * 1000

        idx = df.index[-1]
        if hasattr(idx, "tzinfo"):
            # tz-aware datetime index
            ts = int(idx.timestamp() * 1000)
            return ts
        else:
            # naive datetime index
            ts = int(time.mktime(idx.timetuple()) * 1000)
            return ts
    except Exception:
        return None

def _timestamp_to_datetime(raw_ts: int) -> pd.Timestamp:
    """Convert milliseconds timestamp to pandas Timestamp in Asia/Taipei timezone."""
    # NOTE: 兼容秒(10位)或毫秒(13位)時間戳
    unit = "ms" if int(raw_ts) > 10_000_000_000 else "s"
    dt = pd.to_datetime(int(raw_ts), unit=unit, utc=True).tz_convert("Asia/Taipei")
    return dt

class TradingBot:
    """Trading bot entry point (daemon mode).

    You asked to keep other modules untouched. Therefore:
    - WS listening / reconnect / dedupe / backfill are handled here + utils modules.
    - The *main trading logic* remains in `run()` for you to implement.

    Runtime contract for `run()`
    ----------------------------
    When daemon triggers a run, the following fields are prepared:

      - self.current_trigger_close_ts_ms : int
          The bar close time (ms) that triggered this run (idempotency key).

      - self.history_df : pandas.DataFrame
          Latest fetched OHLCV+FNG dataframe (length = lookback_bars).

    You can implement `run()` using those prepared fields without changing
    the call signature.
    """

    def __init__(self, api_key_config: APIKeyConfig, config: dict):
        self.config = config
        self.api_key = api_key_config.api_key
        self.api_secret = api_key_config.api_secret
        self.discord_bot_token = api_key_config.discord_bot_token
        self.discord_channel_id = api_key_config.discord_channel_id

        self.logger = logging.getLogger(self.__class__.__name__)

        # Operational files
        self.daemon_cfg = self.config.get("daemon", {}) or {}
        self.pidfile = self.daemon_cfg.get("pidfile", "./runtime/trader.pid")
        self.state_store = BotStateStore(self.daemon_cfg.get("state_path", "./runtime/trader_state.json"))

        # Daemon trigger interval (WS stream interval)
        self.trigger_interval = str(self.daemon_cfg.get("trigger_interval", "1h"))
        self.trigger_interval_ms = _parse_interval_to_ms(self.trigger_interval)
        self.run_mode: Literal['STATE_ONLY',"LIVE_TRADE"] = "STATE_ONLY"  # "STATE_ONLY" or "LIVE_TRADE" : run() 狀態控管, 區分 backfill or live trade
        self.strategy_history_df = None  # 用於 run() 的歷史資料 DataFrame

        # Runtime fields prepared for user-implemented run()
        self.current_trigger_close_ts_ms: Optional[int] = None
        self.history_df = None

        self._init_all_modules()
        self.logger.info("All modules initialized successfully.")

        # Daemon state (idempotency)
        self._last_processed_ms: Optional[int] = self.state_store.load_last_processed()
        self._run_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    # -----------------------------
    # Initialization Methods
    # -----------------------------
    def _init_all_modules(self):
        """初始化所有模組"""
        self.data_collector = self._init_exchange_data_collector()
        self.feature_computer = self._init_feature_computer()
        self.long_predictor, self.short_predictor = self._init_long_short_predictor()
        self.strategy_manager = self._init_strategy_manager()
        self.trader = self._init_trader()
        self._init_discord_notifier()

        # Strategy Manager Bootstrap
        self.lookback_bars = (
            self.feature_cfg.get("feat_normalization", {}).get("rolling_window", 144)
            + self.predictor_cfg.get("seq_len", 144)
        )
        self.barTimeframe = self.feature_cfg.get("time", {}).get("freq", "15m")
        self.symbol = self.config.get("symbol", "BTCUSDT")
        self.logger.info(
            "Initialized with symbol=%s, barTimeframe=%s, lookback_bars=%d, trigger_interval=%s",
            self.symbol,
            self.barTimeframe,
            self.lookback_bars,
            self.trigger_interval,
        )

        strategy_warmup_df = self.data_collector.fetch_ohlcv_fng(self.symbol, self.strategy_timeframe, self.strategy_lookback_bars)
        self.logger.debug("Fetched history_df with shape=%s", getattr(strategy_warmup_df, "shape", None))
        self.strategy_manager.bootstrap_from_df(history_df=strategy_warmup_df)

    def _init_exchange_data_collector(self) -> ExchangeDataCollector:
        self.exchange_cfg = self.config.get("exchange", None)
        if not self.exchange_cfg:
            raise ValueError("Exchange configuration not found in config file.")

        self.exchange_cfg = ExchangeConfig(**self.exchange_cfg)
        return ExchangeDataCollector(self.api_key, self.api_secret, self.exchange_cfg)

    def _init_feature_computer(self) -> FeatureComputer:
        feature_cfg_path = Path(self.config["feature"].get("config_path", None))
        if not feature_cfg_path.exists():
            raise FileNotFoundError("Feature configuration path not found in config file.")
        self.feature_cfg = feature_cfg = load_config(feature_cfg_path)
        return FeatureComputer(feature_cfg)

    def _init_strategy_manager(self):
        strategy_cfg_path = Path(self.config["strategy"].get("config_path", None))
        if not strategy_cfg_path.exists():
            raise FileNotFoundError("Strategy configuration path not found in config file.")
        strategy_cfg = load_config(strategy_cfg_path)["SNRStrategy"]
        self.strategy_cfg = SNRCfg(**strategy_cfg)
        self.strategy_timeframe = self.strategy_cfg.timeframe
        self.strategy_lookback_bars = self.strategy_cfg.lookback_bars
        return SNRLiveStrategy(self.strategy_cfg)

    def _init_long_short_predictor(self):
        predictor_cfg_path = Path(self.config["predictor"].get("config_path", None))
        if not predictor_cfg_path.exists():
            raise FileNotFoundError("Predictor configuration path not found in config file.")
        self.predictor_cfg = load_config(predictor_cfg_path)
        self.predictor_bars = self.predictor_cfg.get("seq_len", 144)
        return Predictor(self.predictor_cfg, "long"), Predictor(self.predictor_cfg, "short")

    def _init_trader(self):
        trader_cfg = self.config.get("trade", {})
        return Trader(
            tradeConfig=trader_cfg,
            exchangeConfig=self.exchange_cfg,
            apiKey=self.api_key,
            apiSecret=self.api_secret,
        )
    def _init_discord_notifier(self):
        self.discord_notifier = DiscordNotifier(self.discord_bot_token, self.discord_channel_id, self.logger)
        self.logger.info("Discord Notifier initialized (pending async connect).")

    # -----------------------------
    # User-owned Trading Logic
    # -----------------------------
    def run(self):
        """Main trading logic (TO IMPLEMENT).

        When daemon triggers a run, you can use:

          - self.current_trigger_close_ts_ms
          - self.history_df

        Suggested flow (you already described):
          1) strategy update on latest bar
          2) if signal -> compute features -> predict -> risk checks -> execute
          3) logging/notify

        Notes:
        - Do NOT update self._last_processed_ms here. That is handled by the daemon
          only if this method finishes successfully.
        - Raise exceptions if you want the daemon to treat this run as failed and
          stop backfilling further bars.
        limitation:
                因為你是用 to_thread(self.run)：

        run() 內不要呼叫 asyncio API（因為它跑在 thread，不在 event loop）

        run() 內要把錯誤丟出（raise）或清楚回傳失敗，讓 _run_once_with_finalize_retry 能判斷是否要 commit state

        你在 run() 裡可以放心使用：

        你現有的 data_collector/feature/strategy/predictor/trader（都是同步）

        logging

        DB / discord（同步）
        但如果 discord/DB 是 async client，建議另做同步 wrapper 或改成在 event loop 端排隊處理。
        """
        # TODO: implement the run logic
        if self.strategy_history_df is None or self.strategy_history_df.empty:
            raise ValueError("strategy_history_df is not prepared by daemon layer.")
        
        if self.current_trigger_close_ts_ms is None:
            raise ValueError("current_trigger_close_ts_ms is not set.")
        elif (
            self.current_trigger_close_ts_ms is not None
            and self.strategy_history_df["kline_close_timestamp_ms"].iloc[-1] < self.current_trigger_close_ts_ms
        ):
            raise ValueError("strategy_history_df does not contain data up to current_trigger_close_ts_ms.")

        
        latest_candle = Candle(
            close_time=_timestamp_to_datetime(self.strategy_history_df["kline_close_timestamp_ms"].iloc[-1]),
            open=self.strategy_history_df["open"].iloc[-1],
            high=self.strategy_history_df["high"].iloc[-1],
            low=self.strategy_history_df["low"].iloc[-1],
            close=self.strategy_history_df["close"].iloc[-1],
            volume=self.strategy_history_df["volume"].iloc[-1],
        )
        #回合結束:  Remove strategy_history_df
        self.strategy_history_df = None
        
        if self.run_mode == "STATE_ONLY":
            self.strategy_manager.on_candle_close(latest_candle)
            self.logger.info(f"[STATE_ONLY] Strategy state updated for candle close_time={latest_candle.close_time}")
            self.discord_notifier.send_info(f"[STATE_ONLY] Strategy state updated for candle close_time={latest_candle.close_time}")
            return
        elif self.run_mode == "LIVE_TRADE":
            signals = self.strategy_manager.on_candle_close(latest_candle)
            if not signals:
                self.logger.info(f"[LIVE_TRADE] No trading signals generated for candle close_time={latest_candle.close_time}")
                return
            model_df = self.data_collector.fetch_ohlcv_fng(self.symbol, self.barTimeframe, self.lookback_bars)

            # signal deduplication
            before_dedup_count = len(signals)
            unique_trigger_times = set()
            signals = [sig for sig in signals if sig['test_trigger_time'] not in unique_trigger_times and not unique_trigger_times.add(sig['test_trigger_time'])]
            after_dedup_count = len(signals)
            if after_dedup_count < before_dedup_count:
                self.logger.warning(f"[LIVE_TRADE] Deduplicated signals: before={before_dedup_count}, after={after_dedup_count}")

            for signal in signals:
                signal_type = signal.get("signal_type", "UNKNOWN")
                consume_time = None
                can_long_entry, can_short_entry = False, False
                prediction_success = False
                if signal_type == "Long":
                    try:
                        long_feature = self.feature_computer.compute(model_df,"long")
                        consume_time, can_long_entry  = self.long_predictor.predict(long_feature.iloc[-self.predictor_bars:])
                        prediction_success = True
                    except Exception as e:
                        self.logger.exception(f"[LIVE_TRADE] Long prediction failed for candle close_time={latest_candle.close_time}: {e}")
                        
                elif signal_type == "Short":
                    try:
                        short_feature = self.feature_computer.compute(model_df,"short")
                        consume_time , can_short_entry = self.short_predictor.predict(short_feature.iloc[-self.predictor_bars:])
                        prediction_success = True
                    except Exception as e:
                        self.logger.exception(f"[LIVE_TRADE] Short prediction failed for candle close_time={latest_candle.close_time}: {e}")   
                else:
                    self.logger.warning(f"[LIVE_TRADE] Unknown signal type: {signal_type}")
                    continue

                # update signal with prediction results for logging/notification
                signal['prediction'] = can_long_entry if signal_type == "Long" else can_short_entry

                # 下單邏輯，如果 predict 出錯， fallback 回基礎信號，先以 log 處理
                if (signal_type == "Long"):
                    if (can_long_entry):
                        self.logger.info(f"[LIVE_TRADE] Executing Long trade for candle close_time={latest_candle.close_time} (prediction time: {consume_time:.2f}s)")
                    elif(not can_long_entry and prediction_success):
                        self.logger.info(f"[LIVE_TRADE] Long prediction did not allow entry for candle close_time={latest_candle.close_time} (prediction time: {consume_time:.2f}s)")
                    else:
                        self.logger.warning(f"[LIVE_TRADE] Long signal generated but prediction failed for candle close_time={latest_candle.close_time}; no entry executed.")
                elif (signal_type == "Short"):
                    if (can_short_entry):
                        self.logger.info(f"[LIVE_TRADE] Executing Short trade for candle close_time={latest_candle.close_time} (prediction time: {consume_time:.2f}s)")
                    elif(not can_short_entry and prediction_success):
                        self.logger.info(f"[LIVE_TRADE] Short prediction did not allow entry for candle close_time={latest_candle.close_time} (prediction time: {consume_time:.2f}s)")
                    else:
                        self.logger.warning(f"[LIVE_TRADE] Short signal generated but prediction failed for candle close_time={latest_candle.close_time}; no entry executed.")
            # Send signals to Discord in batch
            if self.discord_notifier and signals:
                try:
                    self.discord_notifier.send_signals_in_batch(signals, self.exchange_cfg.id)
                except Exception as e:
                    self.logger.exception(f"Failed to send signals batch to Discord: {e}")
        return


        raise NotImplementedError("Implement TradingBot.run() using self.history_df and self.current_trigger_close_ts_ms")

    # -----------------------------
    # Daemon Mode (WS-triggered)
    # -----------------------------
    def stop(self) -> None:
        """
        Request daemon stop (used by signal handler).

        """
        self._stop_event.set()

    async def run_forever(self):
        self.logger.info("Starting TradingBot daemon...")
        if hasattr(self, "discord_notifier") and self.discord_notifier is not None:
            try:
                await self.discord_notifier.connect()
                self.logger.info("Discord Notifier connected.")
                await self.discord_notifier.send_info("TradingBot initialized and connected to Discord channel.")
            except Exception:
                self.logger.exception("Discord Notifier connect failed; continue without Discord notifications.")
        listener = BinanceFuturesKlineListener(
            symbol=self.symbol,
            interval=self.trigger_interval,
            testnet=True,
            logger=self.logger,
        )
        listener.start()

        loop = asyncio.get_running_loop()
        self._install_signal_handlers(loop)

        agen = listener.closed_kline_events()

        # NOTE: 不要用 wait_for 包 __anext__；timeout 會 cancel __anext__，可能導致 async generator 終止
        next_evt_task = asyncio.create_task(agen.__anext__())
        stop_task = asyncio.create_task(self._stop_event.wait())

        try:
            while True:
                done, pending = await asyncio.wait(
                    {next_evt_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if stop_task in done:
                    self.logger.info("Stop event set; shutting down daemon loop...")
                    # best-effort cancel pending next event
                    if not next_evt_task.done():
                        next_evt_task.cancel()
                    break

                if next_evt_task in done:
                    try:
                        evt = next_evt_task.result()
                    except StopAsyncIteration:
                        self.logger.warning("Listener event stream ended (StopAsyncIteration).")
                        break
                    except Exception:
                        self.logger.exception("Listener event stream crashed.")
                        break

                    # prepare next event task BEFORE handling current (avoid gaps)
                    next_evt_task = asyncio.create_task(agen.__anext__())

                    if self._stop_event.is_set():
                        self.logger.info("Stop requested; skip handling new event.")
                        break

                    await self._on_kline_closed(evt)

        finally:
            # cleanup tasks
            try:
                if not stop_task.done():
                    stop_task.cancel()
            except Exception:
                pass
            try:
                if next_evt_task and not next_evt_task.done():
                    next_evt_task.cancel()
            except Exception:
                pass

            await listener.stop()
            if hasattr(self, "discord_notifier") and self.discord_notifier is not None:
                try:
                    self.discord_notifier.send_info("TradingBot is shutting down. Goodbye!")
                    await self.discord_notifier.close()
                    self.logger.info("Discord Notifier closed.")
                except Exception:
                    self.logger.exception("Failed to close Discord Notifier.")
            self.logger.info("TradingBot daemon stopped.")


    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        安装 SIGINT (通常是 control + C)/SIGTERM(systemd/docker/OS kill) 处理程序以优雅地停止守护进程。
        """
        def _on_signal():
            self.logger.info("Signal received, stopping daemon...")
            self.stop()

        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(s, _on_signal)
            except NotImplementedError:
                # Windows / limited event loops
                pass

    async def _on_kline_closed(self, evt: KlineClosedEvent) -> None:
        """Handle a single 'kline closed' event.
        處理單個“kline closed”事件。

        Notes
        -----
        - We do not assume WS events are unique or ordered.
        - Dedupe is performed against persisted last_processed timestamp.
        """
        bar_close_ts_ms = int(evt.bar_close_ts_ms)
        # lock : 保證單次運行
        async with self._run_lock:
            # Dedupe: skip duplicates or older triggers (reconnect can re-deliver).
            if self._last_processed_ms is not None and bar_close_ts_ms <= self._last_processed_ms:
                self.logger.info("Skip bar ts=%d (last=%d)", bar_close_ts_ms, self._last_processed_ms)
                return

            await self._backfill_and_run_until(bar_close_ts_ms)

    async def _backfill_and_run_until(self, target_close_ts_ms: int) -> None:
        """Backfill missing triggers and run sequentially up to target_close_ts_ms.
        處理缺失的觸發器並依次運行直到 target_close_ts_ms。
        Implementation approach
        ----------------------
        We base idempotency on WS trigger close timestamps, not on dataframe index,
        so your pipeline can use any barTimeframe (e.g., 15m) while still triggering
        hourly. This keeps the daemon robust and decoupled from dataframe schema.

        Behavior
        --------
        - If last_processed is None: run only the target trigger.
        - Else: run all missing triggers at fixed interval steps (1h by default),
          then run the target trigger.
        - If a run fails, we stop the backfill chain to avoid state gaps.
        """
        to_run: List[int] = []

        if self._last_processed_ms is None:
            to_run = [target_close_ts_ms]
        else:
            last = self._last_processed_ms
            # Generate missing close timestamps based on trigger interval.
            ts = last + self.trigger_interval_ms
            while ts <= target_close_ts_ms:
                to_run.append(ts)
                ts += self.trigger_interval_ms

        if not to_run:
            return

        for ts in to_run:
            if ts < target_close_ts_ms:
                self.run_mode = "STATE_ONLY"  # backfill mode
            elif ts == target_close_ts_ms:
                self.run_mode = "LIVE_TRADE"  # live trade mode
            else:
                self.logger.error(f"In Backfill, ts and close ts_ms mismatch: ts={ts}, target_close_ts_ms={target_close_ts_ms}; datetime={_timestamp_to_datetime(ts)} vs {_timestamp_to_datetime(target_close_ts_ms)}; this should not happen.")
                break
            ok = await self._run_once_with_finalize_retry(ts)
            if not ok:
                self.logger.warning("Run failed at ts=%d datetime=%s; stop backfill chain", ts, _timestamp_to_datetime(ts))
                break

            # Commit idempotency only after successful run.
            self._last_processed_ms = ts
            self.state_store.save_last_processed(ts)

    async def _run_once_with_finalize_retry(self, trigger_close_ts_ms: int) -> bool:
        """Run one cycle with REST-finalize retries.

        Why
        ---
        WS close event can arrive slightly before REST endpoints expose the finalized
        candle / merged dataset. We do a short retry loop to wait until REST data
        appears to be "fresh enough".

        Success criteria
        ----------------
        - We can fetch history_df successfully.
        - (Best-effort) the latest timestamp in history_df is >= trigger_close_ts_ms.

        If your collector does not expose comparable timestamps, the criteria falls
        back to "fetch succeeds".
        """
        max_retry = int(self.daemon_cfg.get("finalize_max_retry", 5))
        sleep_sec = float(self.daemon_cfg.get("finalize_retry_sleep_sec", 2.0))

        for attempt in range(1, max_retry + 1):
            if self._stop_event.is_set():
                return False

            try:
                strategy_history_df = self.data_collector.fetch_ohlcv_fng(self.symbol, self.strategy_timeframe, self.strategy_lookback_bars)
                latest_ts = _extract_latest_ts_ms(strategy_history_df)
                # 1000 ms 容差，避免毫秒級誤差導致誤判，若 latest_ts 明顯落後於 trigger，代表 REST 還沒 finalize，就等一下再抓
                if latest_ts is not None and latest_ts + 1000 < trigger_close_ts_ms:
                    self.logger.info(
                        "REST not finalized yet (need=%d latest=%d) attempt=%d/%d",
                        trigger_close_ts_ms,
                        latest_ts,
                        attempt,
                        max_retry,
                    )
                    await asyncio.sleep(sleep_sec)
                    continue

                # Prepare runtime fields for user-implemented run()
                # 截斷 DataFrame 至 trigger_close_ts_ms, 保證策略狀態更新一致性
                self.current_trigger_close_ts_ms = int(trigger_close_ts_ms)
                self.strategy_history_df = strategy_history_df[
                    strategy_history_df["kline_close_timestamp_ms"] <= trigger_close_ts_ms
                ].copy()
                self.logger.debug(
                    "截斷測試 (%s): current_trigger_close_ts_ms=%s, strategy_history_df latest close ts=%s",
                    trigger_close_ts_ms == strategy_history_df["kline_close_timestamp_ms"].iloc[-1],
                    self.current_trigger_close_ts_ms,
                    _extract_latest_ts_ms(self.strategy_history_df),
                )

                # Execute user logic in a thread to keep event loop responsive.
                await asyncio.to_thread(self.run)

                return True

            except NotImplementedError:
                # Surface the error clearly; treat as failure so state doesn't advance.
                self.logger.exception("TradingBot.run() is not implemented.")
                return False
            except Exception:
                
                self.logger.exception(
                    "run_once failed: ts=%d datetime=%s attempt=%d/%d",
                    trigger_close_ts_ms,
                    _timestamp_to_datetime(trigger_close_ts_ms),
                    attempt,
                    max_retry,
                )
                await asyncio.sleep(sleep_sec)

        return False


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)

    # File logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=log_dir / "trader.log",
        filemode="a",
    )

    # Also print INFO+ to console for easier debugging (optional but practical).
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger("").addHandler(console)
    # NOTE: silence noisy websockets DEBUG logs; keep our bot logs at DEBUG.
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("websockets.client").setLevel(logging.INFO)



def main():
    api_key_config = APIKeyConfig(
        api_key=os.getenv("TRADER_API_KEY"),
        api_secret=os.getenv("TRADER_API_SECRET"),
        discord_bot_token=os.getenv("DISCORD_BOT_TOKEN"),
        discord_channel_id=int(os.getenv("DISCORD_CHANNEL_ID")),
    )   

    parser = argparse.ArgumentParser(description="Trader Application")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store log files")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["daemon", "init"],
        default="daemon",
        help="daemon: run forever (WS-triggered); init: only initialize modules",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    setup_logging(log_dir)

    config_path = args.config
    config = load_config(config_path)

    bot = TradingBot(api_key_config, config)

    if args.mode == "init":
        bot.logger.info("Init mode: modules initialized; exiting.")
        return

    guard = ProcessGuard(bot.pidfile)
    guard.acquire()
    try:
        asyncio.run(bot.run_forever())
    finally:
        guard.release()


if __name__ == "__main__":
    main()
