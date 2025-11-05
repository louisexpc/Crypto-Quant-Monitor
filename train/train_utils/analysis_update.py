# ./app/analysis.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# ★ 新增：可選用 DBSCAN 作價格帶聚類
#   若你的環境未安裝 sklearn，請 pip install scikit-learn
try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


class Level:
    """
    用於表示和追蹤支撐/阻力水平狀態的數據類。
    這個物件將在市場模擬過程中動態更新。（沿用原設計）
    """
    def __init__(self, price: float, level_type: str, snr_type: str, created_at):
        self.price = price
        self.type = level_type  # 'support' or 'resistance' - 可變
        self.snr_type = snr_type # 'SNR1' or 'SNR2' - 不變

        # 狀態追蹤
        self.is_valid = True
        # created_at 明確定義為：產生該 SNR 水平的「後一根 K 線」的時間
        self.created_at: pd.Timestamp = created_at
        self.flipped_at: Optional[pd.Timestamp] = None
        self.last_tested_at: Optional[pd.Timestamp] = None

    def __repr__(self):
        return (f"Level(price={self.price}, type='{self.type}', snr='{self.snr_type}', "
                f"valid={self.is_valid}, flipped_at={self.flipped_at})")


# ★ 新增：Band 結構（對外 API 不暴露，內部使用）
class _Band:
    """
    聚合多個相近 Level 形成的價格帶（band）。
    - center: 中心價（中位數或加權平均）
    - width: 半寬（建議 = alpha_w * ATR）
    - core_ratio: 核心帶比例，core = [center - beta_core*width, center + beta_core*width]
    - type: 'support' 或 'resistance'（由成員 Level 多數決）
    - score: 用於排序的強度（觸碰次數、最近度、量能等可擴充）
    - members: 該帶包含的 level 索引
    - flipped_at: 帶級「翻轉」的最近時間（當帶內 Level 首次達成帶級 Flip 時更新）
    - last_tested_at: 最近一次 wick test 的時間
    - created_at: 第一個成員的 created_at 最小值（帶的起源）
    """
    def __init__(self,
                 center: float,
                 width: float,
                 level_types: List[str],
                 member_indices: List[int],
                 created_ats: List[pd.Timestamp],
                 beta_core: float = 0.4):
        self.center = center
        self.width = width
        self.core_ratio = beta_core
        self.type = self._majority_type(level_types)
        self.members = list(member_indices)
        self.score: float = 1.0
        self.flipped_at: Optional[pd.Timestamp] = None
        self.last_tested_at: Optional[pd.Timestamp] = None
        self.created_at: pd.Timestamp = min(created_ats) if created_ats else None

    @property
    def low(self) -> float:
        return self.center - self.width

    @property
    def high(self) -> float:
        return self.center + self.width

    @property
    def core_low(self) -> float:
        return self.center - self.core_ratio * self.width

    @property
    def core_high(self) -> float:
        return self.center + self.core_ratio * self.width

    @staticmethod
    def _majority_type(types: List[str]) -> str:
        # support / resistance 多數決；平手則保留第一個
        if not types:
            return 'support'
        s = sum(1 for t in types if t == 'support')
        r = len(types) - s
        return 'support' if s >= r else 'resistance'

    def __repr__(self):
        return (f"Band(center={self.center:.2f}, width={self.width:.2f}, "
                f"type='{self.type}', core=({self.core_low:.2f},{self.core_high:.2f}), "
                f"members={len(self.members)})")


class StrategyAnalyzerUpdate:
    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        volume_ema_window: int = 10,
        # ★ 新增：Band 模式開關與參數（預設關閉 → 保持與舊版行為一致）
        use_band: bool = False,
        band_alpha_w: float = 0.25,      # w = alpha_w * ATR
        band_beta_core: float = 0.4,     # core 比例 → core = [c - beta*w, c + beta*w]
        band_gamma_flip: float = 0.3,    # flip 緩衝比例：close 必須超過邊界 + gamma*w
        dbscan_min_samples: int = 3,     # DBSCAN 每群最少點數
        dbscan_eps_in_atr: float = 0.35, # DBSCAN eps = (dbscan_eps_in_atr * ATR)
        band_use_median_center: bool = True  # True: 中位數；False:（簡單）平均
    ):
        """
        ★ 新增參數說明（band_* 與 dbscan_*）：
        - use_band: 是否啟用「價格帶」判定。False=沿用舊版單一水平；True=使用帶級 Flip/Test。
        - band_alpha_w: 帶半寬的 ATR 倍數。建議 0.2 ~ 0.35，依 timeframe 與資產波動校準。
        - band_beta_core: 核心帶比例，影線觸及 core 更可靠（降噪）。
        - band_gamma_flip: Flip 緩衝比例，避免剛越過邊界就當作 Flip。
        - dbscan_min_samples: DBSCAN 聚類之最小成員數，防止微弱孤立點成帶。
        - dbscan_eps_in_atr: 用 ATR 尺度化 DBSCAN 的半徑。例：0.35→eps=0.35*ATR。
        - band_use_median_center: True 用中位數作為帶中心（較穩健）。
        """
        if df.empty or len(df) < 2:
            raise ValueError("DataFrame for analysis must contain at least 2 candles.")

        # --------------------------------------------------------------
        # ★ 修正段落 ★ 排除未收線 K 線 → 僅保留最後一根已收線
        self.df = df.copy()
        self.symbol = symbol
        self.timeframe = timeframe
        self.levels: List[Level] = []
        self.signals: List[Dict[str, Any]] = []

        _unit_map = {'m': 'minutes', 'h': 'hours', 'd': 'days'}
        _duration = None
        for _u, _kw in _unit_map.items():
            if timeframe.endswith(_u):
                _duration = pd.Timedelta(**{_kw: int(timeframe.rstrip(_u))})
                break

        if _duration is not None and len(self.df) >= 2:
            now_ts = pd.Timestamp.utcnow()
            if self.df.index[-1] + _duration > now_ts:
                self.df = self.df.iloc[:-1]

        if len(self.df) < 2:
            raise ValueError("DataFrame after trimming incomplete candle must contain at least 2 candles.")
        # --------------------------------------------------------------

        # 成交量 EMA（沿用）
        if 'volume' not in self.df.columns:
            raise ValueError("DataFrame must contain 'volume' column for volume EMA calculation.")
        self.volume_ema_window = volume_ema_window
        self.df['volume_ema'] = self.df['volume'].ewm(span=self.volume_ema_window, adjust=False).mean()

        # ★ 新增：ATR（for 帶寬與 DBSCAN 度量）
        #   需要 'high','low','close' 欄位
        for col in ['high', 'low', 'close', 'open']:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column.")
        self.df['tr'] = np.maximum(
            self.df['high'] - self.df['low'],
            np.maximum((self.df['high'] - self.df['close'].shift(1)).abs(),
                       (self.df['low'] - self.df['close'].shift(1)).abs())
        )
        # 使用典型 14 期 ATR，也可視需要調參
        self.df['ATR'] = self.df['tr'].rolling(window=14, min_periods=1).mean()

        # 仍採用最後一根 *已收線* 判斷
        self.last_candle = self.df.iloc[-1]

        # ★ Band 模式開關與參數（保存在物件中）
        self.use_band = use_band
        self.band_alpha_w = float(band_alpha_w)
        self.band_beta_core = float(band_beta_core)
        self.band_gamma_flip = float(band_gamma_flip)
        self.dbscan_min_samples = int(dbscan_min_samples)
        self.dbscan_eps_in_atr = float(dbscan_eps_in_atr)
        self.band_use_median_center = bool(band_use_median_center)

        # ★ 內部狀態：在 simulate 後，由 levels 聚成 bands
        self._bands: List[_Band] = []  # 僅 use_band=True 時會生成


    # ------------------------ 市場演進（沿用） ------------------------
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
        """根據兩根相鄰的 K 線，判斷是否形成新的 SNR1 或 SNR2 水平（沿用，前置過濾 volume>EMA）。"""
        # 先停用量能過濾
        # if prev_candle['volume'] < prev_candle['volume_ema'] or current_candle['volume'] < current_candle['volume_ema']:
        #     return
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
        """根據當前 K 線，更新所有已存在水平的狀態（測試、突破、轉換、失效）（沿用）。"""
        for level in self.levels:
            # (1) 失效水平重新檢驗
            if not level.is_valid:
                is_broken_through_body = (
                    (level.type == 'resistance' and candle['close'] > level.price) or
                    (level.type == 'support'    and candle['close'] < level.price)
                )
                if is_broken_through_body:
                    level.is_valid = True
                    # 重新驗證的K線不算作測試，等下一根
                    continue

            if not level.is_valid:
                continue

            open_price, high, low, close = (
                candle['open'], candle['high'], candle['low'], candle['close']
            )

            # 2-a. 實體完整突破 → Flip
            is_broken_up = (
                level.type == 'resistance' and open_price < level.price < close
            )
            is_broken_down = (
                level.type == 'support'    and open_price > level.price > close
            )
            if is_broken_up or is_broken_down:
                level.type = 'support' if is_broken_up else 'resistance'
                level.flipped_at = candle.name
                level.is_valid = True
                continue  # 翻轉後不再檢查影線測試

            # 2-b. 影線觸及 + 實體未跨越 → Test
            body_min, body_max = sorted([open_price, close])
            same_side_body = not (body_min <= level.price <= body_max)
            is_tested_by_wick = (low <= level.price <= high) and same_side_body

            if is_tested_by_wick:
                level.last_tested_at = candle.name
                level.is_valid = False


    # ------------------------ ★ 新增：band 建構與判定工具 ------------------------
    def _build_bands_from_levels(self):
        """
        ★ 將 self.levels 聚為價格帶：
          1) 取最後一段資料的 ATR（用 last_candle 對應 ATR 或 rolling 略平滑）
          2) 以 eps = dbscan_eps_in_atr * ATR 做 DBSCAN 聚類
          3) 每群形成 _Band（center 使用 median 或 mean；width=alpha_w*ATR）
          4) 帶 type 以成員 Level 多數決決定
        """
        if not self.use_band:
            self._bands = []
            return

        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for band-mode (DBSCAN). Please install it or set use_band=False.")

        if len(self.levels) == 0:
            self._bands = []
            return

        # 取當前 ATR（可切換為更平滑的尾部平均）
        atr_now = float(self.df['ATR'].iloc[-1])
        if atr_now <= 0 or np.isnan(atr_now):
            # 若 ATR 不可用，fallback：用歷史收斂均值
            atr_now = float(self.df['ATR'].dropna().tail(50).mean())
            if np.isnan(atr_now) or atr_now <= 0:
                # 實在無法計算時，設為價格的 0.2% 作為尺度（保底）
                atr_now = float(self.last_candle['close']) * 0.002

        eps = max(1e-9, self.dbscan_eps_in_atr * atr_now)  # 避免 0
        X = np.array([[lv.price] for lv in self.levels], dtype=float)
        db = DBSCAN(eps=eps, min_samples=self.dbscan_min_samples).fit(X)
        labels = db.labels_

        bands: List[_Band] = []
        for k in sorted(set(labels)):
            if k == -1:
                # 噪音點略過（也可考慮：孤立點直接轉為小帶）
                continue
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                continue

            prices = [self.levels[i].price for i in idx]
            types = [self.levels[i].type for i in idx]
            created_list = [self.levels[i].created_at for i in idx]

            center = (np.median(prices) if self.band_use_median_center
                      else float(np.mean(prices)))
            width = max(1e-9, self.band_alpha_w * atr_now)  # 半寬

            band = _Band(center=center,
                         width=width,
                         level_types=types,
                         member_indices=idx.tolist(),
                         created_ats=created_list,
                         beta_core=self.band_beta_core)

            # ★ 簡單 score：成員數 + 最近度（可自由擴充）
            #   你可以根據成員的最近測試時間、量能等去加權
            band.score = len(idx)
            bands.append(band)

        # 依 score 降序（強帶優先）
        bands.sort(key=lambda b: b.score, reverse=True)
        self._bands = bands


    @staticmethod
    def _flipped_up_band(candle: pd.Series, c: float, w: float, gamma: float) -> bool:
        """
        ★ 帶級「向上 Flip」判定：
        - 開在中心下方、收在「上邊界 + gamma*w」之上（保留緩衝防剛越界）
        """
        return (candle['open'] < c) and (candle['close'] > (c + gamma * w))

    @staticmethod
    def _flipped_down_band(candle: pd.Series, c: float, w: float, gamma: float) -> bool:
        """
        ★ 帶級「向下 Flip」判定：
        - 開在中心上方、收在「下邊界 - gamma*w」之下
        """
        return (candle['open'] > c) and (candle['close'] < (c - gamma * w))

    @staticmethod
    def _wick_test_without_body_cross_band(candle: pd.Series, c: float, w: float, beta: float) -> bool:
        """
        ★ 僅純影線觸及帶（尤其 core）才算「測試」：
        - 影線觸及 core： [c - beta*w, c + beta*w]
        - 實體不跨中心線（避免已反轉）
        """
        core_low, core_high = (c - beta * w, c + beta * w)
        body_min, body_max = sorted([candle['open'], candle['close']])
        wick_hits_core = (candle['low'] <= core_high) and (candle['high'] >= core_low)
        body_not_cross_center = not (body_min <= c <= body_max)
        return wick_hits_core and body_not_cross_center


    # ------------------------ 訊號裁決（延伸：同時支援舊/新） ------------------------
    def _check_last_candle_for_signal_levels(self):
        """
        沿用「單一水平點」的 Flip→Test→Entry（舊邏輯）。
        """
        for level in self.levels:
            was_flipped = level.flipped_at is not None
            if not was_flipped:
                continue

            # 量能過濾
            was_tested_by_last_candle = (
                (level.last_tested_at == self.last_candle.name) and
                (self.last_candle['volume'] >= self.last_candle['volume_ema'])
            )
            if not was_tested_by_last_candle:
                continue

            test_after_flip = level.last_tested_at > level.flipped_at
            if not test_after_flip:
                continue

            signal_type = 'Long' if level.type == 'support' else 'Short'

            signal = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "signal_type": signal_type,
                # 產生該水平的「後一根 K 線」
                "signal_candle_time": level.created_at,
                # 最後測試觸發的 K 線（最新 K 線）
                "test_trigger_time": self.last_candle.name,
                "level_price": level.price,
                "level_current_type": level.type,
                "level_snr_type": level.snr_type,
                "level_flipped_at": level.flipped_at,
                # ★ 附加：保持兼容，不暴露 band 相關
            }
            self.signals.append(signal)


    def _check_last_candle_for_signal_bands(self):
        """
        ★ 帶級 Flip→Test→Entry。
        - 事件鏈：帶級 Flip（由歷史中的某根觸發，記錄 band.flipped_at）→
                 最後一根已收線 wick test core + 量能條件 → 依帶類型入場
        """
        if not self._bands:
            return

        # 掃歷史 K 線，為每個 band 更新 flipped_at / last_tested_at
        # 為效率起見，可只看最近 N 根；這裡保持清晰寫法
        for i in range(1, len(self.df)):
            cndl = self.df.iloc[i]
            for band in self._bands:
                c, w = band.center, band.width
                # 若尚未翻轉，檢查是否達成帶級 Flip
                if band.flipped_at is None:
                    if band.type == 'resistance':
                        if self._flipped_up_band(cndl, c, w, self.band_gamma_flip):
                            # 翻轉：阻力→支撐
                            band.type = 'support'
                            band.flipped_at = cndl.name
                            # 翻轉當根不再檢查 test（避免自相矛盾）
                            continue
                    else:  # band.type == 'support'
                        if self._flipped_down_band(cndl, c, w, self.band_gamma_flip):
                            # 翻轉：支撐→阻力
                            band.type = 'resistance'
                            band.flipped_at = cndl.name
                            continue
                else:
                    # 已翻轉過，可更新最近 wick 測試（不要求每根都檢查量能）
                    if self._wick_test_without_body_cross_band(cndl, c, w, self.band_beta_core):
                        band.last_tested_at = cndl.name

        # 用「最後一根已收線」做裁決：是否 test + 量能 + test 在 flip 之後
        for band in self._bands:
            if band.flipped_at is None:
                continue
            if band.last_tested_at != self.last_candle.name:
                continue
            if not (self.last_candle['volume'] >= self.last_candle['volume_ema']):
                continue
            if not (band.last_tested_at > band.flipped_at):
                continue

            signal_type = 'Long' if band.type == 'support' else 'Short'

            # ★ 訊號輸出保持與舊版兼容的鍵名（level_*），但值寫入 band 中心價等
            signal = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "signal_type": signal_type,
                # 使用帶的起源時間（第一個成員 level 的 created_at 最小值）
                "signal_candle_time": band.created_at,
                "test_trigger_time": self.last_candle.name,
                "level_price": float(band.center),             # 沿用欄位名，值為帶中心
                "level_current_type": band.type,               # 'support'/'resistance'
                "level_snr_type": "BAND",                      # 標註為 BAND
                "level_flipped_at": band.flipped_at,
                # ★ 亦可加上一些可選的帶資訊（不破壞原欄位）：
                "band_low": float(band.low),
                "band_high": float(band.high),
                "band_core_low": float(band.core_low),
                "band_core_high": float(band.core_high),
                "band_score": float(band.score),
                "band_members": len(band.members),
            }
            self.signals.append(signal)


    # ------------------------ 對外主流程 ------------------------
    def analyze(self) -> List[Dict[str, Any]]:
        """
        執行完整的分析流程：先模擬市場，再對最後一根 K 線做判斷。
        - 預設（use_band=False）：沿用舊版「單一水平」邏輯
        - 若 use_band=True：以 ATR+DBSCAN 形成價格帶，套用「帶級 Flip→Test→Entry」
        """
        self._simulate_market_evolution()

        if self.use_band:
            self._build_bands_from_levels()
            self._check_last_candle_for_signal_bands()
        else:
            self._check_last_candle_for_signal_levels()

        return self.signals
