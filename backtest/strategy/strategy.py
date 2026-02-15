# snr_live_strategy.py
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Literal


# ============================================================
# Utilities
# ============================================================

def parse_timeframe_to_timedelta(timeframe: Literal["5m","15m", "30m", "1h", "4h", "1d"]) -> pd.Timedelta:
    """
    將 timeframe 字串轉為 pandas Timedelta。
    支援:
      - "5m","15m", "30m"
      - "1h", "4h"
      - "1d"

    Args:
      - timeframe: 時間框架字串

    Return:
      - return : pd.Timedelta
    """
    _unit_map = {"m": "minutes", "h": "hours", "d": "days"}
    for u, kw in _unit_map.items():
        if timeframe.endswith(u):
            n = int(timeframe.rstrip(u))
            return pd.Timedelta(**{kw: n})
    raise ValueError(f"Unsupported timeframe format: {timeframe}. Expected like '5m'/'15m'/'1h'/'1d'.")


# ============================================================
# Data Structures
# ============================================================

@dataclass(frozen=True)
class Candle:
    """
    Production feed candle（已收線事件才會送入策略）

    注意：
      - open_time: K 線開盤時間（保留欄位，用於 debug / 回測對齊）
      - close_time: K 線收盤時間（本次改動後的唯一識別與比較基準）
    """
    close_time: pd.Timestamp 
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_time: Optional[pd.Timestamp] = None
    


@dataclass(frozen=True)
class CandleState:
    """
    Candle + indicator state（用於做到與回測版零偏差）
    - vol_ema: 該根 candle close 當下的 volume EMA 值
    """
    candle: Candle
    vol_ema: float


class Level:
    """
    用於表示和追蹤支撐/阻力水平狀態的數據類。
    這個物件將在市場模擬過程中動態更新。
    """
    def __init__(self, price: float, level_type: str, snr_type: str, created_at):
        self.price = price
        self.type = level_type   # 'support' or 'resistance' - 可變
        self.snr_type = snr_type # 'SNR1' or 'SNR2' - 不變

        # 狀態追蹤
        self.is_valid = True
        # created_at 明確定義為：產生該 SNR 水平的「後一根 K 線」的時間（= current candle close_time）
        # NOTE: 2026-01 - 對齊 data engine：以 close_time 作為唯一識別
        self.created_at: pd.Timestamp = created_at
        self.flipped_at: Optional[pd.Timestamp] = None
        self.last_tested_at: Optional[pd.Timestamp] = None

    def __repr__(self):
        return (f"Level(price={self.price}, type='{self.type}', snr='{self.snr_type}', "
                f"valid={self.is_valid}, flipped_at={self.flipped_at})")


# ============================================================
# EMA (incremental)
# ============================================================

class EMAState:
    """
    增量 EMA 狀態容器（避免 production 每次重算整段 df）

    - alpha = 2/(span+1)
    - ema_t = alpha * x_t + (1-alpha) * ema_{t-1}
    """
    def __init__(self, span: int):
        if span <= 0:
            raise ValueError("EMA span must be positive.")
        self.span = span
        self.ema: Optional[float] = None

    def update(self, x: float) -> float:
        """
        更新 EMA

        Args:
          - x: 最新觀測值（此處用 volume）

        Return:
          - return : 更新後的 EMA 值
        """
        alpha = 2.0 / (self.span + 1.0)
        if self.ema is None:
            self.ema = float(x)
        else:
            self.ema = alpha * float(x) + (1.0 - alpha) * self.ema
        return self.ema


# ============================================================
# Production-ready Strategy (event-driven)
# ============================================================
@dataclass(frozen=True)
class SNRCfg:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    lookback_bars: int = 100
    volume_ema_window: int = 10
    max_levels: int = 5000
    dedup: Dict[str, Any] = None
    execution:Dict[str, Any] = None

class SNRLiveStrategy:
    """
    將你現行 StrategyAnalyzer（回測型）改寫成 production 常駐版本：
    - 事件驅動：只在 candle closed 時呼叫 on_candle_close()
    - 增量更新：每次只處理 prev + current 一組，不重跑全歷史
    - 去重：避免重連 replay 同一根 candle 造成重複處理
    - bootstrap：啟動時可用近期歷史 K warmup 狀態（不 emit signals）

    重要：本版本已做到「與回測版零偏差」的 volume gating：
    - 產生新水平需要：prev.volume >= prev.vol_ema AND curr.volume >= curr.vol_ema
    - signal 需要：curr.volume >= curr.vol_ema（且 wick test/flip 條件成立）
    """

    def __init__(self, cfg: SNRCfg):
        """
        初始化策略（production 常駐）

        Args:
          - cfg: 對應 YAML 的 dict（建議傳入 cfg["SNRStrategy"] 區塊）
          - symbol: 交易標的
          - timeframe: 此策略 instance 綁定的 timeframe（例如 "15m"）

        Return:
          - return : None
        """
        self.cfg = cfg
        self.symbol = cfg.symbol
        self.timeframe = cfg.timeframe
        self.tf_delta = parse_timeframe_to_timedelta(cfg.timeframe)

        # -------------------------
        # YAML Config
        # -------------------------

        self.volume_ema_window = cfg.volume_ema_window
        self.vol_ema = EMAState(span=self.volume_ema_window)

        self.max_levels = cfg.max_levels

        self.dedup = cfg.dedup
        self.dedup_enabled: bool = self.dedup.get("enabled", True) if self.dedup else False

        self.execution = cfg.execution
        self.cooldown_bars: int = self.execution.get("cooldown_bars", 0) if self.execution else 0

        # -------------------------
        # Runtime States
        # -------------------------
        self.levels: List[Level] = []
        self.signals: List[Dict[str, Any]] = []  # 可選：保留歷史 emitted signals

        # 用 CandleState 保留「該根 close 當下的 vol_ema」，以做到零偏差 gating
        self.prev_state: Optional[CandleState] = None

        # 去重：本次改用 candle 收盤時間（close_time）作為唯一識別（對齊 data engine / WS trigger）
        # NOTE: 變數名稱維持不改（最小修改），但其語意已改為「last processed close_time」。
        self._last_processed_close_time: Optional[pd.Timestamp] = None

        # dedup emit：避免 replay 同一根 candle 的重複 emit
        self._last_emitted_key: Optional[Tuple] = None

        # cooldown：以 bar counter 計數（只在 target timeframe instance 內）
        self._bar_counter: int = 0
        self._cooldown_until_bar: int = -1

    # ============================================================
    # Bootstrap
    # ============================================================

    def bootstrap_from_df(self, history_df: pd.DataFrame) -> None:
        """
        使用近期歷史已收線 K 初始化狀態（不 emit signals）

        Args:
          - history_df: OHLCV DataFrame
                        - index: candle open datetime (Asia/Taipei)
                        - columns: kline_open_timestamp_ms/kline_close_timestamp_ms/open/high/low/close/volume
                                - kline_close_timestamp_ms (ms): close 時間戳記

        Return:
          - return : None
        """
        if history_df is None or history_df.empty:
            return

        df = history_df.sort_index()

        for _open_datetime, row in df.iterrows():
            candle = Candle(
                close_time=pd.Timestamp(row["kline_close_timestamp_ms"], unit="ms", tz="Asia/Taipei"),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                open_time=pd.Timestamp(row["kline_open_timestamp_ms"], unit="ms", tz="Asia/Taipei"),
            )
            self._process_candle_close(candle, emit_signal=False)

    # ============================================================
    # Live Event Entrypoint
    # ============================================================

    def on_candle_close(self, candle: Candle) -> List[Dict[str, Any]]:
        """
        production 主入口：只在 candle closed 時呼叫（你已保證事件特性）

        Args:
          - candle: 已收線 Candle（策略判斷/去重以 close_time 為準）

        Return:
          - return : List[Dict[str, Any]]
            - 本次 candle close 觸發的 signals（可能為空 list）
        """
        return self._process_candle_close(candle, emit_signal=True)

    # ============================================================
    # Core Incremental Pipeline
    # ============================================================

    def _process_candle_close(self, candle: Candle, emit_signal: bool) -> List[Dict[str, Any]]:
        """
        核心增量流程（每根 candle close 呼叫一次）

        Args:
          - candle: 已收線 Candle
          - emit_signal: 是否輸出 signal（bootstrap 時設 False）

        Return:
          - return : List[Dict[str, Any]]（此 candle 觸發的 signals）
        """
        # -------- (0) idempotency / dedup processing --------
        if self.dedup_enabled:
            # NOTE: 2026-01 - 對齊 data engine：以 close_time 作為去重與順序比較基準
            candle_close_time = candle.close_time
            if self._last_processed_close_time is not None and candle_close_time <= self._last_processed_close_time:
                return []
            self._last_processed_close_time = candle_close_time

        self._bar_counter += 1

        # -------- (1) update volume EMA and build CandleState --------
        curr_vol_ema = self.vol_ema.update(candle.volume)
        curr_state = CandleState(candle=candle, vol_ema=curr_vol_ema)

        # -------- (2) need prev_state to form SNR pair --------
        if self.prev_state is None:
            self.prev_state = curr_state
            return []

        prev_state = self.prev_state
        self.prev_state = curr_state

        # -------- (3) identify new levels (ZERO-BIAS gating) --------
        self._identify_new_levels(prev_state, curr_state)

        # -------- (4) update existing levels --------
        self._update_existing_levels(candle)

        # -------- (5) bound levels size --------
        if len(self.levels) > self.max_levels:
            self.levels = self.levels[-self.max_levels:]

        # -------- (6) emit signals (if enabled) --------
        if not emit_signal:
            return []

        if self.cooldown_bars > 0 and self._bar_counter <= self._cooldown_until_bar:
            return []

        signals = self._check_signal_on_candle(curr_state)

        # dedup emit：避免同一根 candle 被重放造成重複 signal
        if self.dedup_enabled and signals:
            sig0 = signals[0]
            # NOTE: 2026-01 - key 改用 close_time，避免 open_time 與 WS trigger key 不一致
            key = (candle.close_time, sig0.get("level_price"), sig0.get("signal_type"))
            if self._last_emitted_key == key:
                return []
            self._last_emitted_key = key

        if signals and self.cooldown_bars > 0:
            self._cooldown_until_bar = self._bar_counter + self.cooldown_bars

        self.signals.extend(signals)
        return signals

    # ============================================================
    # Strategy Logic (ported; with ZERO-BIAS volume gating)
    # ============================================================

    def _identify_new_levels(self, prev_state: CandleState, curr_state: CandleState) -> None:
        """
        根據兩根相鄰的 K 線，判斷是否形成新的 SNR1 或 SNR2 水平。
        本版本嚴格回到回測版 gating：
          prev.volume >= prev.vol_ema AND curr.volume >= curr.vol_ema 才允許產生新水平。

        Args:
          - prev_state: 前一根 CandleState（含 prev.vol_ema）
          - curr_state: 當前 CandleState（含 curr.vol_ema）

        Return:
          - return : None
        """
        prev = prev_state.candle
        curr = curr_state.candle

        # 0905 Update: 僅在成交量高於其 EMA 時，才允許產生新的水平
        if (prev.volume < prev_state.vol_ema) or (curr.volume < curr_state.vol_ema):
            return

        prev_dir = 1 if prev.close > prev.open else -1
        curr_dir = 1 if curr.close > curr.open else -1

        prev_body = abs(prev.close - prev.open)
        curr_body = abs(curr.close - curr.open)

        # SNR1: 相反方向 + body 相近
        if prev_dir == -curr_dir and abs(prev_body - curr_body) <= max(prev_body, curr_body) * 0.25:
            price = (prev.close + curr.open) / 2
            level_type = "resistance" if prev_dir == 1 else "support"
            self.levels.append(Level(price, level_type, "SNR1", curr.close_time))  # NOTE: 2026-01 - created_at 改用 close_time

        # SNR2: 同方向（且非 0）
        if prev_dir == curr_dir and curr_dir != 0:
            price = (prev.close + curr.open) / 2
            level_type = "support" if curr_dir == 1 else "resistance"
            self.levels.append(Level(price, level_type, "SNR2", curr.close_time))  # NOTE: 2026-01 - created_at 改用 close_time

    def _update_existing_levels(self, candle: Candle) -> None:
        """
        根據當前 K 線，更新所有已存在水平的狀態（測試、突破、轉換、失效）。
        （邏輯保持與原始版本一致，含 re-validate bugfix 與 wick test 修正）

        Args:
          - candle: 當前已收線 candle

        Return:
          - return : None
        """
        for level in self.levels:
            # ------------- (1) 失效水平重新檢驗 -------------
            if not level.is_valid:
                is_broken_through_body = (
                    (level.type == "resistance" and candle.close > level.price) or
                    (level.type == "support"    and candle.close < level.price)
                )
                if is_broken_through_body:
                    level.is_valid = True
                    # Bug Fix (2023-08-23): 重新驗證的K線不算作測試
                    continue

            if not level.is_valid:
                continue

            # ------------- (2) 解析當前 K 線 -------------
            open_price, high, low, close = candle.open, candle.high, candle.low, candle.close

            # ------- 2-a. 判斷實體是否「完整」突破水平 → Flip -------
            is_broken_up = (level.type == "resistance" and open_price < level.price < close)
            is_broken_down = (level.type == "support"    and open_price > level.price > close)

            if is_broken_up or is_broken_down:
                level.type = "support" if is_broken_up else "resistance"
                level.flipped_at = candle.close_time  # NOTE: 2026-01 - flipped_at 改用 close_time
                level.is_valid = True
                continue

            # ------- 2-b. 僅純影線觸及才算「測試」 -------
            body_min, body_max = sorted([open_price, close])
            same_side_body = not (body_min <= level.price <= body_max)
            is_tested_by_wick = (low <= level.price <= high) and same_side_body

            if is_tested_by_wick:
                level.last_tested_at = candle.close_time  # NOTE: 2026-01 - last_tested_at 改用 close_time
                level.is_valid = False

    def _check_signal_on_candle(self, curr_state: CandleState) -> List[Dict[str, Any]]:
        """
        最終 candle 裁決（production：每根 close 都裁決一次）

        Args:
          - curr_state: 當前 CandleState（含 curr.vol_ema）

        Return:
          - return : List[Dict[str, Any]]
            - 可能為空；可能包含多個 level 的 signal（視你的 levels 狀態）
        """
        candle = curr_state.candle
        curr_vol_ema = curr_state.vol_ema
        out: List[Dict[str, Any]] = []

        for level in self.levels:
            was_flipped = level.flipped_at is not None
            if not was_flipped:
                continue

            # 0910 Update: 最後一根 K 線交易量必須高於其 EMA
            was_tested_by_this_candle = (
                level.last_tested_at == candle.close_time and  # NOTE: 2026-01 - 以 close_time 判斷是否為本根測試
                candle.volume >= curr_vol_ema
            )
            if not was_tested_by_this_candle:
                continue

            # 測試必須發生在 flip 之後
            if level.last_tested_at is None or level.flipped_at is None:
                continue
            test_after_flip = level.last_tested_at > level.flipped_at
            if not test_after_flip:
                continue

            signal_type = "Long" if level.type == "support" else "Short"

            signal = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "signal_type": signal_type,
                # Signal Candle：產生該 SNR 水平的「後一根 K 線」的時間
                "signal_candle_time": level.created_at,
                # Test Trigger Candle：完成最後「測試」動作的 K 線（即當前 K 線）
                "test_trigger_time": candle.close_time,  # NOTE: 2026-01 - 改用 close_time
                "level_price": level.price,
                "level_current_type": level.type,
                "level_snr_type": level.snr_type,
                # Flip Candle：水平發生翻轉時的 K 線
                "level_flipped_at": level.flipped_at,
                # Debug: volume gating visibility
                "volume": candle.volume,
                "volume_ema": curr_vol_ema,
            }
            out.append(signal)

        return out
