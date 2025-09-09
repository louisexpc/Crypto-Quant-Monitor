# ./app/analysis.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class Level:
    """
    用於表示和追蹤支撐/阻力水平狀態的數據類。
    這個物件將在市場模擬過程中動態更新。
    """
    def __init__(self, price: float, level_type: str, snr_type: str, created_at):
        self.price = price
        self.type = level_type  # 'support' or 'resistance' - 可變
        self.snr_type = snr_type # 'SNR1' or 'SNR2' - 不變
        
        # 狀態追蹤
        self.is_valid = True
        # created_at 現在明確定義為：產生該 SNR 水平的「後一根 K 線」的時間
        self.created_at: pd.Timestamp = created_at
        self.flipped_at: Optional[pd.Timestamp] = None
        self.last_tested_at: Optional[pd.Timestamp] = None

    def __repr__(self):
        return (f"Level(price={self.price}, type='{self.type}', snr='{self.snr_type}', "
                f"valid={self.is_valid}, flipped_at={self.flipped_at})")

class StrategyAnalyzer:
    def __init__(self, df: pd.DataFrame, symbol: str, timeframe: str, volume_ema_window: int = 10):
        if df.empty or len(df) < 2:
            raise ValueError("DataFrame for analysis must contain at least 2 candles.")

        # --------------------------------------------------------------
        # ★ 修正段落 ★
        # 目的：**排除尚未收線的當前 K 線**（避免用 04:00‑04:15 這種未完成
        #       的 K 線來判斷訊號），只保留最後一根「已收線」的 K 線。
        # 作法：
        #   1. 解析 timeframe（僅支援 m/h/d 單位），取得一個 Timedelta。
        #   2. 若最後一筆 K 線的 index + duration 仍晚於現在，
        #      視為「還在形成中」，將其移除。
        #   3. 後續所有邏輯都使用裁剪後的 self.df。
        # --------------------------------------------------------------
        self.df = df.copy()  # 不直接改動外部傳入的 DataFrame
        self.symbol = symbol
        self.timeframe = timeframe
        self.levels: List[Level] = []
        self.signals: List[Dict[str, Any]] = []

        # -------- 解析 timeframe，例：'15m' → 15 分鐘 --------
        _unit_map = {'m': 'minutes', 'h': 'hours', 'd': 'days'}
        _duration = None
        for _u, _kw in _unit_map.items():
            if timeframe.endswith(_u):
                _duration = pd.Timedelta(**{_kw: int(timeframe.rstrip(_u))})
                break

        # -------- 移除未收線 K 線 --------
        if _duration is not None and len(self.df) >= 2:
            now_ts = pd.Timestamp.utcnow()
            # candle.index = 開盤時間，收盤時間 = index + duration
            if self.df.index[-1] + _duration > now_ts:
                # 尚未收線：捨棄最後一筆
                self.df = self.df.iloc[:-1]

        # 再次確認資料量充足
        if len(self.df) < 2:
            raise ValueError("DataFrame after trimming incomplete candle must contain at least 2 candles.")

        # -------- 仍採用「最後一根 *已收線*」作為判斷基準 --------
        self.last_candle = self.df.iloc[-1]
        # --------------------------------------------------------------

        # 0905 Update:
        # 新增 volume_ema_window 參數，預設10
        # 最為 level 產生時，考量的條件之一，來針對訊號做 denoise
        # 初步想法為：僅在成交量高於該 EMA 時，才允許產生訊號
        if 'volume' not in self.df.columns:
            raise ValueError("DataFrame must contain 'volume' column for volume EMA calculation.")
        self.volume_ema_window = volume_ema_window
        self.df['volume_ema'] = self.df['volume'].ewm(span=self.volume_ema_window, adjust=False).mean()


    def _simulate_market_evolution(self):
        """
        階段一：模擬市場演進。
        從歷史數據開始，逐根 K 線推演，動態建立和更新所有水平的狀態。
        這個過程只更新狀態，不產生訊號。
        """
     
        
        for i in range(1, len(self.df)):
            prev_candle = self.df.iloc[i-1]
            current_candle = self.df.iloc[i]
            
            self._identify_new_levels(prev_candle, current_candle)
            self._update_existing_levels(current_candle)

    
    def _identify_new_levels(self, prev_candle: pd.Series, current_candle: pd.Series):
        """根據兩根相鄰的 K 線，判斷是否形成新的 SNR1 或 SNR2 水平。"""
        # 0905 Update: 僅在成交量高於其 EMA 時，才允許產生新的水平
        if prev_candle['volume'] < prev_candle['volume_ema'] or current_candle['volume'] < current_candle['volume_ema']:
            return
        prev_dir = 1 if prev_candle['close'] > prev_candle['open'] else -1
        curr_dir = 1 if current_candle['close'] > current_candle['open'] else -1
        
        prev_body = abs(prev_candle['close'] - prev_candle['open'])
        curr_body = abs(current_candle['close'] - current_candle['open'])

        if prev_dir == -curr_dir and abs(prev_body - curr_body) <= max(prev_body, curr_body) * 0.25:
            price = (prev_candle['close'] + current_candle['open']) / 2
            level_type = 'resistance' if prev_dir == 1 else 'support'
            self.levels.append(Level(price, level_type, 'SNR1', current_candle.name))

        if prev_dir == curr_dir and curr_dir != 0:
            price = (prev_candle['close'] + current_candle['open']) / 2
            level_type = 'support' if curr_dir == 1 else 'resistance'
            self.levels.append(Level(price, level_type, 'SNR2', current_candle.name))


    def _update_existing_levels(self, candle: pd.Series):
        """根據當前 K 線，更新所有已存在水平的狀態（測試、突破、轉換、失效）。"""
        for level in self.levels:
            # ------------- (1) 失效水平重新檢驗 -------------
            if not level.is_valid:
                is_broken_through_body = (
                    (level.type == 'resistance' and candle['close'] > level.price) or
                    (level.type == 'support'    and candle['close'] < level.price)
                )
                if is_broken_through_body:
                    level.is_valid = True

                    # Bug Fix (2023-08-23):
                    # [EN] Previously, when a level was re-validated (i.e., a previously invalid level became valid again),
                    # the same candle could immediately trigger a test or flip, which was incorrect.
                    # To ensure proper validation, after a level is re-validated, we skip further test/flip logic for this candle.
                    # The candle that re-validates the level does NOT count as a test.
                    # [ZH] 修正：當失效的水平被重新驗證有效時，必須等到下一根K線才允許進行 test 或 flip。
                    # 重新驗證的K線不算作測試，避免邏輯錯誤。
                    continue  

            if not level.is_valid:
                continue  # 尚未被重新驗證，跳過其餘邏輯

            # ------------- (2) 解析當前 K 線 -------------
            open_price, high, low, close = (
                candle['open'], candle['high'], candle['low'], candle['close']
            )

            # ------- 2‑a. 判斷實體是否「完整」突破水平 → Flip -------
            is_broken_up = (
                level.type == 'resistance' and open_price < level.price < close
            )
            is_broken_down = (
                level.type == 'support'    and open_price > level.price > close
            )
            if is_broken_up or is_broken_down:
                # Flip：阻力→支撐 / 支撐→阻力
                level.type = 'support' if is_broken_up else 'resistance'
                level.flipped_at = candle.name
                level.is_valid = True
          
                continue  # 翻轉後不再檢查影線測試

            # ------- 2‑b. 僅純影線觸及才算「測試」 -------
            # ★ 修正處 ★：需排除「實體穿過」水平的情況
            body_min, body_max = sorted([open_price, close])          # 實體上下界
            same_side_body = not (body_min <= level.price <= body_max)  # True → 實體未跨越  # <-- 修正
            is_tested_by_wick = (low <= level.price <= high) and same_side_body             # <-- 修正

            if is_tested_by_wick:
                level.last_tested_at = candle.name
                level.is_valid = False



    def _check_last_candle_for_signal(self):
        """
        階段二：最終 K 線裁決。
        僅評估最新完成的 K 線，判斷它是否觸發了任何進場訊號。
        """

        for level in self.levels:
            was_flipped = level.flipped_at is not None
            if not was_flipped:
                continue

            was_tested_by_last_candle = (level.last_tested_at == self.last_candle.name)
            if not was_tested_by_last_candle:
                continue
            
            test_after_flip = level.last_tested_at > level.flipped_at
            if not test_after_flip:
                continue

            signal_type = 'Long' if level.type == 'support' else 'Short'
            

            
            # --- 關鍵修正：調整訊號字典結構以符合新定義 ---
            signal = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "signal_type": signal_type,
                # Signal Candle：產生該 SNR 水平的「後一根 K 線」的時間
                "signal_candle_time": level.created_at,
                # Test Trigger Candle：完成最後「測試」動作的 K 線（即最新 K 線）
                "test_trigger_time": self.last_candle.name,
                "level_price": level.price,
                "level_current_type": level.type,
                "level_snr_type": level.snr_type,
                # Flip Candle：水平發生翻轉時的 K 線
                "level_flipped_at": level.flipped_at,
            }
            self.signals.append(signal)

    def analyze(self) -> List[Dict[str, Any]]:
        """
        執行完整的分析流程：先模擬市場，再對最後一根 K 線做判斷。
        """

        
        self._simulate_market_evolution()
        self._check_last_candle_for_signal()
        
        return self.signals