# features/indicators.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Union, Iterable, List
import re
import numpy as np
import pandas as pd
import pandas_ta as ta
from tsfeatures import tsfeatures as _tsf  # Nixtla 版 API：tsfeatures(panel, freq=...)
try:
    from train.train_utils.random_alpha_generator.random_alpha_generator import load_alpha
except ImportError:  # pragma: no cover - tqdm optional
    pass

from tqdm.auto import tqdm

# =========================
# 1) 指標庫：負責把已規格化的 df → 各種特徵
# =========================
class IndicatorLibrary:
    """
    僅負責計算特徵：給定已正規化的 df（DatetimeIndex、OHLCV、附帶其他欄位）後，提供 builders。
    """

    def __init__(self, df: pd.DataFrame):
        """
        df: 已正規化、對齊時間索引的 DataFrame；需包含 open/high/low/close/volume。
        """
        need = ["open", "high", "low", "close", "volume"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"IndicatorLibrary 缺少必要欄位: {missing}")
        self.df = df
        self.ohlcv = {
            "open":   df["open"],
            "high":   df["high"],
            "low":    df["low"],
            "close":  df["close"],
            "volume": df["volume"],
        }


    @staticmethod
    def _resolve_sign_mode(sign) -> str:
        """
        1. 說明: 將 sign 參數標準化為 'cont'（連續）、'sign'（符號化）或 'both'。
        2. inputs:
           - sign(bool|str): True/'sign' → 'sign'；False/'cont' → 'cont'；'both' → 'both'。
        3. return: str，於 {'cont','sign','both'} 之一。
        """
        if isinstance(sign, str):
            s = sign.strip().lower()
            if s in ("both",): return "both"
            if s in ("true","y","1","sign"): return "sign"
            if s in ("false","n","0","cont","continuous"): return "cont"
            raise ValueError(f"Invalid sign param: {sign}")
        return "sign" if sign is True else "cont"

    @staticmethod
    def _iter_params(val) -> List[int]:
        """允許單一數值或可迭代的數值，回傳整數列表。"""
        if isinstance(val, (list, tuple, set)):
            return [int(x) for x in val]
        return [int(val)]

    # ========== Trend / Averages ==========
    def build_SMA(self, length: int | Iterable[int], sign=False) -> pd.DataFrame:
        """
        1. 說明: 計算簡單移動平均（SMA），可輸出連續值與符號版（價格高於/低於 SMA）。
        2. inputs:
           - length(int): SMA 期數。
           - sign(bool|str): 是否輸出符號特徵（'sign'/'cont'/'both'）。
        3. return: DataFrame，欄名 'SMA_{length}' 與/或 'SSMA_{length}'。
        """
        mode = self._resolve_sign_mode(sign)
        out = {}
        for l in self._iter_params(length):
            sma = ta.sma(self.ohlcv["close"], length=l)
            if mode in ("cont","both"):
                out[f"SMA_{l}"] = sma.astype(np.float32)
            if mode in ("sign","both"):
                sig = np.where(self.ohlcv["close"].values > sma.values, 1.0,
                               np.where(self.ohlcv["close"].values < sma.values, -1.0, 0.0))
                out[f"SSMA_{l}"] = pd.Series(sig, index=self.df.index, dtype=np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_EMA(self, length: int | Iterable[int], sign=False) -> pd.DataFrame:
        """
        1. 說明: 計算指數移動平均（EMA），可輸出連續值與符號版。
        2. inputs:
           - length(int): EMA 期數。
           - sign(bool|str): 是否輸出符號特徵。
        3. return: DataFrame，欄名 'EMA_{length}' 與/或 'SEMA_{length}'。
        """
        mode = self._resolve_sign_mode(sign)
        out = {}
        for l in self._iter_params(length):
            ema = ta.ema(self.ohlcv["close"], length=l)
            if mode in ("cont","both"):
                out[f"EMA_{l}"] = ema.astype(np.float32)
            if mode in ("sign","both"):
                sig = np.where(self.ohlcv["close"].values > ema.values, 1.0,
                               np.where(self.ohlcv["close"].values < ema.values, -1.0, 0.0))
                out[f"SEMA_{l}"] = pd.Series(sig, index=self.df.index, dtype=np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_TEMA(self, length: int | Iterable[int], sign=False) -> pd.DataFrame:
        """
        1. 說明: 計算三重指數移動平均（TEMA），可輸出連續值與符號版。
        2. inputs:
           - length(int): TEMA 期數。
           - sign(bool|str): 是否輸出符號特徵。
        3. return: DataFrame，欄名 'TEMA_{length}' 與/或 'STEMA_{length}'。
        """
        mode = self._resolve_sign_mode(sign)
        out = {}
        for l in self._iter_params(length):
            tema = ta.tema(self.ohlcv["close"], length=l)
            if mode in ("cont","both"):
                out[f"TEMA_{l}"] = tema.astype(np.float32)
            if mode in ("sign","both"):
                sig = np.where(self.ohlcv["close"].values > tema.values, 1.0,
                               np.where(self.ohlcv["close"].values < tema.values, -1.0, 0.0))
                out[f"STEMA_{l}"] = pd.Series(sig, index=self.df.index, dtype=np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_MACD(self, fast: int = 12, slow: int = 26, signal: int = 9, sign=False) -> pd.DataFrame:
        """
        1. 說明: 計算 MACD（主線、訊號線），可輸出主線或符號差（主線-訊號線 的符號）。
        2. inputs:
           - fast(int), slow(int), signal(int): MACD 參數。
           - sign(bool|str): 是否輸出符號特徵（主線-訊號線 > 0 ?）。
        3. return: DataFrame，欄名 'MACD_{f}_{s}_{sig}' 與/或 'SMACD_{...}'。
        """
        mode = self._resolve_sign_mode(sign)
        macd = ta.macd(self.ohlcv["close"], fast=fast, slow=slow, signal=signal)
        cols = list(macd.columns)
        col_macd = [c for c in cols if c.startswith("MACD_") and "MACDh" not in c and "MACDs" not in c]
        col_sig  = [c for c in cols if c.startswith("MACDs_") or c.lower().endswith("signal")]
        macd_line = macd[col_macd[0]] if col_macd else macd.iloc[:,0]
        sig_line  = macd[col_sig[0]]  if col_sig  else (macd.iloc[:,-1] if macd.shape[1]>1 else macd.iloc[:,0])

        out = {}
        if mode in ("cont","both"):
            out[f"MACD_{fast}_{slow}_{signal}"] = macd_line.astype(np.float32)
        if mode in ("sign","both"):
            diff = (macd_line - sig_line).values
            sgn  = np.where(diff>0, 1.0, np.where(diff<0, -1.0, 0.0))
            out[f"SMACD_{fast}_{slow}_{signal}"] = pd.Series(sgn, index=self.df.index, dtype=np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_SLOPE(self, length: int | Iterable[int]) -> pd.DataFrame:
        """
        1. 說明: 計算價格斜率（線性回歸斜率或等價定義，依 pandas_ta）。
        2. inputs:
           - length(int): 回看期。
        3. return: DataFrame，欄名 'SLOPE_{length}'。
        """
        out = {}
        for l in self._iter_params(length):
            s = ta.slope(self.ohlcv["close"], length=l)
            out[f"SLOPE_{l}"] = s.astype(np.float32)
        return pd.DataFrame(out, index=self.df.index)
    
    def build_TTM_TRND(self, length: int = 6) -> pd.DataFrame:
        """
        1. 說明: 計算 TTM Trend（以高低收派生），僅取第一欄為代表。
        2. inputs:
           - length(int): 期數（預設 6）。
        3. return: DataFrame，欄名 'TTM_TRND_{length}'。
        """
        tt = ta.ttm_trend(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], length=length)
        s  = (tt.iloc[:, 0] if isinstance(tt, pd.DataFrame) else tt)
        s  = pd.Series(s, index=getattr(s, "index", self.df.index)).reindex(self.df.index).astype(np.float32)
        return s.rename(f"TTM_TRND_{length}").to_frame()

    def build_DPO(self, length: int | Iterable[int]) -> pd.DataFrame:
        """
        1. 說明: 計算去趨勢振盪（DPO）。
        2. inputs:
           - length(int): 期數。
        3. return: DataFrame，欄名 'DPO_{length}'。
        """
        out = {}
        for l in self._iter_params(length):
            s = ta.dpo(self.ohlcv["close"], length=l, centered=False)
            out[f"DPO_{l}"] = s.astype(np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_AMATE_LR(self, fast: int, slow: int, mamode: int = 2) -> pd.DataFrame:
        """
        1. 說明: 計算 AMAT 指標，回傳其中的 LR（linearity-related）分量。
        2. inputs:
           - fast(int), slow(int): 快慢期參數。
           - mamode(int): 移動平均模式（依 pandas_ta）。
        3. return: DataFrame，欄名 'AMATe_LR_{fast}_{slow}_{mamode}'。
        """
        amat = ta.amat(self.ohlcv["close"], fast=fast, slow=slow, mamode=mamode)
        col = [c for c in amat.columns if "AMATe_LR" in c or "LR_" in c]
        tgt = col[0] if col else amat.columns[0]
        return pd.DataFrame({f"AMATe_LR_{fast}_{slow}_{mamode}": amat[tgt].astype(np.float32)}, index=self.df.index)

    # ========== Momentum / Oscillator ==========
    def build_RSI(self, length: int | Iterable[int]) -> pd.DataFrame:
        """
        1. 說明: 計算 RSI。
        2. inputs:
           - length(int): 期數。
        3. return: DataFrame，欄名 'RSI_{length}'。
        """
        out = {}
        for l in self._iter_params(length):
            s = ta.rsi(self.ohlcv["close"], length=l)
            out[f"RSI_{l}"] = s.astype(np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_MOM(self, length: int | Iterable[int]) -> pd.DataFrame:
        """
        1. 說明: 計算動能（MOM）。
        2. inputs:
           - length(int): 期數。
        3. return: DataFrame，欄名 'MOM_{length}'。
        """
        out = {}
        for l in self._iter_params(length):
            s = ta.mom(self.ohlcv["close"], length=l)
            out[f"MOM_{l}"] = s.astype(np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_STOCH(self, k: int, d: int = 3, smooth_k: int = 3) -> pd.DataFrame:
        """
        1. 說明: 計算隨機指標（STOCH），回傳 %K、%D。
        2. inputs:
           - k(int), d(int), smooth_k(int): 參數設定。
        3. return: DataFrame，欄名 'STOCHk_{k}', 'STOCHd_{k}'。
        """
        st = ta.stoch(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], k=k, d=d, smooth_k=smooth_k)
        st = st.rename(columns={st.columns[0]: f"STOCHk_{k}", st.columns[1]: f"STOCHd_{k}"})
        return st[[f"STOCHk_{k}", f"STOCHd_{k}"]].astype(np.float32)

    def build_KDJ(self, k: int, d: int, smooth_k: int) -> pd.DataFrame:
        """
        1. 說明: 由 STOCH 衍生 KDJ，輸出 J 線（3K - 2D）。
        2. inputs:
           - k(int), d(int), smooth_k(int): 參數設定（同 STOCH）。
        3. return: DataFrame，欄名 'J_{k}_{d}'。
        """
        st = ta.stoch(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], k=k, d=d, smooth_k=smooth_k)
        kcol, dcol = st.columns[0], st.columns[1]
        j = 3 * st[kcol] - 2 * st[dcol]
        return pd.DataFrame({f"J_{k}_{d}": j.astype(np.float32)}, index=self.df.index)

    def build_UO(self, fast: int, medium: int, slow: int) -> pd.DataFrame:
        """
        1. 說明: 計算終極震盪指標（UO）。
        2. inputs:
           - fast(int), medium(int), slow(int): 期數參數。
        3. return: DataFrame，欄名 'UO_{fast}_{medium}_{slow}'。
        """
        s = ta.uo(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], fast=fast, medium=medium, slow=slow)
        return pd.DataFrame({f"UO_{fast}_{medium}_{slow}": s.astype(np.float32)}, index=self.df.index)

    def build_RVI(self, length: int) -> pd.DataFrame:
        """
        1. 說明: 計算相對波動指數（RVI），兼容 pandas_ta 回傳 DataFrame/Series 的差異。
        2. inputs:
           - length(int): 期數。
        3. return: DataFrame，欄名 'RVI_{length}'。
        """
        rvi = ta.rvi(self.ohlcv["close"], length=length)
        if isinstance(rvi, pd.DataFrame):
            col = [c for c in rvi.columns if c.startswith("RVI")]
            tgt = col[0] if col else rvi.columns[0]
            s = rvi[tgt]
        else:
            s = rvi
        return pd.DataFrame({f"RVI_{length}": s.astype(np.float32)}, index=self.df.index)

    def build_CCI(self, length: int, c: float = 0.015) -> pd.DataFrame:
        """
        1. 說明: 計算 CCI。
        2. inputs:
           - length(int): 期數。
           - c(float): 常數比例（預設 0.015）。
        3. return: DataFrame，欄名 'CCI_{length}_{c}'。
        """
        s = ta.cci(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], length=length, c=c)
        return pd.DataFrame({f"CCI_{length}_{c}": s.astype(np.float32)}, index=self.df.index)

    def build_ZS(self, length: int | Iterable[int]) -> pd.DataFrame:
        """
        1. 說明: 計算 z-score（以 close）。
        2. inputs:
           - length(int): 滑動視窗大小。
        3. return: DataFrame，欄名 'ZS_{length}'。
        """
        out = {}
        for l in self._iter_params(length):
            s = ta.zscore(self.ohlcv["close"], length=l)
            out[f"ZS_{l}"] = s.astype(np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_WILLR(self, length: int) -> pd.DataFrame:
        """
        1. 說明: 計算 Williams %R。
        2. inputs:
           - length(int): 期數。
        3. return: DataFrame，欄名 'WILLR_{length}'。
        """
        s = ta.willr(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], length=length)
        return pd.DataFrame({f"WILLR_{length}": s.astype(np.float32)}, index=self.df.index)

    # ========== Volatility ==========
    def build_TRUERANGE(self) -> pd.DataFrame:
        """
        1. 說明: 計算真實波幅（TR）。
        2. inputs: 無。
        3. return: DataFrame，欄名 'TRUERANGE_1'。
        """
        prev_close = self.ohlcv["close"].shift(1)
        tr = pd.concat([
            (self.ohlcv["high"] - self.ohlcv["low"]),
            (self.ohlcv["high"] - prev_close).abs(),
            (self.ohlcv["low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        return pd.DataFrame({"TRUERANGE_1": tr.astype(np.float32)}, index=self.df.index)

    def build_RANGE(self, window: int | Iterable[int], pct: bool = True) -> pd.DataFrame:
        """
        1. 說明: 計算高低價區間的平均（可選百分比尺度）。
        2. inputs:
           - window(int): 滑動視窗。
           - pct(bool): True 為 (H-L)/abs(C) 的均值，False 為 (H-L) 均值。
        3. return: DataFrame，欄名 'RANGE_{window}'。
        """
        H, L, C = self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"]
        hl = (H - L)
        out = {}
        for w in self._iter_params(window):
            if pct:
                base = C.abs().replace(0, np.nan)
                s = (hl / base).rolling(w, min_periods=max(1, w//2)).mean()
            else:
                s = hl.rolling(w, min_periods=max(1, w//2)).mean()
            out[f"RANGE_{w}"] = s.astype(np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_ATR(self, length: int = 14, pct: bool = True) -> pd.DataFrame:
        """
        1. 說明: 計算 ATR 與可選的相對 ATR（ATR/|C|）。
        2. inputs:
           - length(int): ATR 期數。
           - pct(bool): 是否輸出相對值（ATRP）。
        3. return: DataFrame，欄名 'ATR_{length}' 與/或 'ATRP_{length}'。
        """
        atr = ta.atr(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], length=length)
        out = {f"ATR_{length}": atr.astype(np.float32)}
        if pct:
            C = self.ohlcv["close"].abs().replace(0, np.nan)
            out[f"ATRP_{length}"] = (atr / C).astype(np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_MASSI(self, fast: int, slow: int) -> pd.DataFrame:
        """
        1. 說明: 計算質量指標（MASS Index）。
        2. inputs:
           - fast(int), slow(int): 快慢期參數。
        3. return: DataFrame，欄名 'MASSI_{fast}_{slow}'。
        """
        s = ta.massi(self.ohlcv["high"], self.ohlcv["low"], fast=fast, slow=slow)
        return pd.DataFrame({f"MASSI_{fast}_{slow}": s.astype(np.float32)}, index=self.df.index)

    def build_BBP(self, length: int, std: float) -> pd.DataFrame:
        """
        1. 說明: 計算布林通道 %B（BBP）。
        2. inputs:
           - length(int): 期數。
           - std(float): 標準差倍數。
        3. return: DataFrame，欄名 'BBP_{length}_{std}'。
        """
        bb = ta.bbands(self.ohlcv["close"], length=length, std=std)
        col = [c for c in bb.columns if c.startswith("BBP_")]
        tgt = col[0] if col else bb.columns[-1]
        return pd.DataFrame({f"BBP_{length}_{std}": bb[tgt].astype(np.float32)}, index=self.df.index)

    def build_EWMRET(self, halflife: int | Iterable[int]) -> pd.DataFrame:
        """
        1. 說明: 計算對數報酬的 EWM 均值與標準差，支援多個半衰期。
        2. inputs:
           - halflife(int|Iterable[int]): 半衰期或其列表。
        3. return: DataFrame，欄名 'EWM_M_{hl}', 'EWM_S_{hl}'。
        """
        px = self.ohlcv["close"].astype(np.float32)
        lr = np.log(px).diff()
        hls = [halflife] if isinstance(halflife, int) else list(halflife)
        out = {}
        for hl in sorted(set(int(h) for h in hls)):
            m = lr.ewm(halflife=hl, adjust=False).mean()
            s = lr.ewm(halflife=hl, adjust=False).std()
            out[f"EWM_M_{hl}"] = m.astype(np.float32)
            out[f"EWM_S_{hl}"] = s.astype(np.float32)
        return pd.DataFrame(out, index=self.df.index)

    # ========== Price-Volume ==========
    def build_PVO(self, fast: int, slow: int, signal: int) -> pd.DataFrame:
        """
        1. 說明: 計算 PVO（Volume-based Oscillator）。
        2. inputs:
           - fast(int), slow(int), signal(int): 參數設定。
        3. return: DataFrame，主要欄位名稱依 pandas_ta 命名。
        """
        pvo = ta.pvo(self.ohlcv["volume"], fast=fast, slow=slow, signal=signal)
        col = [c for c in pvo.columns if c.startswith("PVO_") and c.count("_") >= 2]
        main = col[0] if col else pvo.columns[0]
        return pd.DataFrame({main: pvo[main].astype(np.float32)}, index=self.df.index)

    def build_PVR(self) -> pd.DataFrame:
        """
        1. 說明: 計算 PVR（Price-Volume Ratio 等價量，依 pandas_ta）。
        2. inputs: 無。
        3. return: DataFrame，欄名 'PVR' 或相容欄名。
        """
        s = ta.pvr(self.ohlcv["close"], self.ohlcv["volume"])
        if isinstance(s, pd.Series):
            return pd.DataFrame({"PVR": s.astype(np.float32)}, index=self.df.index)
        return pd.DataFrame({s.columns[0] if len(s.columns) else "PVR": s.iloc[:,0].astype(np.float32)}, index=self.df.index)

    def build_BOP(self) -> pd.DataFrame:
        """
        1. 說明: 計算 BOP（Balance of Power）。
        2. inputs: 無。
        3. return: DataFrame，欄名 'BOP'。
        """
        s = ta.bop(self.ohlcv["open"], self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"])
        return pd.DataFrame({"BOP": s.astype(np.float32)}, index=self.df.index)

    def build_PXVOL(self) -> pd.DataFrame:
        """
        1. 說明: 建立簡單的價量融合特徵（方向強度、logret×量變、方向×成交量）。
        2. inputs: 無（使用 self.ohlcv）。
        3. return: DataFrame，欄名 'DIR_STRENGTH','PXV_LR_VCHG','DIRxVOL'。
        """
        O, H, L, C, V = (self.ohlcv[k].astype(np.float32) for k in ("open","high","low","close","volume"))
        eps = 1e-9
        dir_strength = (C - O) / (np.maximum(H - L, eps))
        logret = np.log(C).diff()
        vol_chg = V.pct_change().fillna(0.0).clip(-1.0, 1.0)
        out = {
            "DIR_STRENGTH":  dir_strength.astype(np.float32),
            "PXV_LR_VCHG":   (logret * vol_chg).astype(np.float32),
            "DIRxVOL":       (dir_strength * V).astype(np.float32),
        }
        return pd.DataFrame(out, index=self.df.index)

    # ========== Returns ==========
    def build_LOGRET(self, lags: Iterable[int], pct: bool = False) -> pd.DataFrame:
        """
        1. 說明: 計算對數報酬（或百分比報酬）的多 lag 欄位。
        2. inputs:
           - lags(Iterable[int]): 落後期數集合。
           - pct(bool): True → pct_change；False → log diff。
        3. return: DataFrame，欄名 'LOGRET_{k}'。
        """
        px = self.ohlcv["close"].astype(float)
        out = {}
        for k in sorted(set(int(x) for x in lags)):
            if pct:
                out[f"LOGRET_{k}"] = px.pct_change(k).astype(np.float32)
            else:
                out[f"LOGRET_{k}"] = np.log(px).diff(k).astype(np.float32)
        return pd.DataFrame(out, index=self.df.index)

    # ========== Time (cyc) ==========
    def build_TIME_CYC(self, tz: str = "UTC", daily: bool = True, weekly: bool = True) -> pd.DataFrame:
        """
        1. 說明: 建立日/週週期性的正餘弦時間特徵（以指定時區）。
        2. inputs:
           - tz(str): 轉換時區（預設 UTC）。
           - daily(bool): 是否輸出日內週期（TOD）。
           - weekly(bool): 是否輸出週期（DOW）。
        3. return: DataFrame，欄名 'TOD_SIN','TOD_COS','DOW_SIN','DOW_COS'（依選項）。
        """
        idx = pd.DatetimeIndex(self.df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx = idx.tz_convert(tz)
        out = {}
        if daily:
            hour = idx.hour + idx.minute / 60.0
            out["TOD_SIN"] = np.sin(2*np.pi*hour/24.0).astype(np.float32)
            out["TOD_COS"] = np.cos(2*np.pi*hour/24.0).astype(np.float32)
        if weekly:
            dow = idx.dayofweek.astype(float)
            out["DOW_SIN"] = np.sin(2*np.pi*dow/7.0).astype(np.float32)
            out["DOW_COS"] = np.cos(2*np.pi*dow/7.0).astype(np.float32)
        return pd.DataFrame(out, index=self.df.index)
    
    # ========== fast (15 min) ==========
    def build_15_DIR(self) -> pd.DataFrame:
        """
        1. 說明: 由 m15_{0..3}_close 產出三段 15m 漲跌指標（01、12、23）。
        2. inputs: 無（需 self.df 內已有 m15_* 欄位）。
        3. return: DataFrame，欄名 'M15_DIR_01','M15_DIR_12','M15_DIR_23'。
        """
        x = self.df
        need = ["m15_0_close","m15_1_close","m15_2_close","m15_3_close"]
        for c in need:
            if c not in x.columns:
                raise ValueError(f"缺少欄位 {c}（請確認 combiner 已產出 m15_* 欄位）")

        d01 = (x["m15_1_close"] > x["m15_0_close"]).astype("float32") \
                .where(x[["m15_0_close","m15_1_close"]].notna().all(axis=1))
        d12 = (x["m15_2_close"] > x["m15_1_close"]).astype("float32") \
                .where(x[["m15_1_close","m15_2_close"]].notna().all(axis=1))
        d23 = (x["m15_3_close"] > x["m15_2_close"]).astype("float32") \
                .where(x[["m15_2_close","m15_3_close"]].notna().all(axis=1))

        return pd.DataFrame({
            "M15_DIR_01": d01,
            "M15_DIR_12": d12,
            "M15_DIR_23": d23,
        }, index=self.df.index)

    def build_15_VOL(self) -> pd.DataFrame:
        """
        1. 說明: 回傳四段 15m 量能（m15_0..3_vol 的直通）。
        2. inputs: 無（需 self.df 內已有對應欄位）。
        3. return: DataFrame，欄名 'M15_VOL_0/1/2/3'。
        """
        x = self.df
        need = ["m15_0_vol","m15_1_vol","m15_2_vol","m15_3_vol"]
        for c in need:
            if c not in x.columns:
                raise ValueError(f"缺少欄位 {c}（請確認 combiner 已產出 m15_* 欄位）")
        return pd.DataFrame({
            "M15_VOL_0": pd.to_numeric(x["m15_0_vol"], errors="coerce").astype("float32"),
            "M15_VOL_1": pd.to_numeric(x["m15_1_vol"], errors="coerce").astype("float32"),
            "M15_VOL_2": pd.to_numeric(x["m15_2_vol"], errors="coerce").astype("float32"),
            "M15_VOL_3": pd.to_numeric(x["m15_3_vol"], errors="coerce").astype("float32"),
        }, index=self.df.index)
    
    def build_fng(self) -> pd.DataFrame:
        """
        1. 說明: 由原始 FNG 欄位計算差分與 7d z-score（不在此處 shift）。
        2. inputs: 無（需 self.df 內已有 'sent_fng'）。
        3. return: DataFrame，欄名 'sent_fng','sent_fng_diff1','sent_fng_z7d'。
        """
        x = self.df
        need = ["fng"]
        missing = [c for c in need if c not in x.columns]
        if missing:
            raise ValueError(
                f"缺少 FNG 欄位：{missing}。請先把 fng_15m_utc.csv 併入主表 "
                f"(或用 merge_fng_into_15m.py) 再呼叫 build_fng。"
            )

        fng = pd.to_numeric(x["fng"], errors="coerce").astype("float32")
        diff1 = fng.diff()
        roll = fng.rolling("7d", min_periods=3)
        z7d = (fng - roll.mean()) / roll.std()

        out = pd.DataFrame({
            "sent_fng":        fng,
            "sent_fng_diff1":  diff1.astype("float32"),
            "sent_fng_z7d":    z7d.astype("float32"),
        }, index=self.df.index)

        return out
    
    def build_alpha(self, paths: Iterable[Union[str, Path]]) -> pd.DataFrame:
        """
        1. 說明: 讀取 save_alpha 輸出的 JSON 檔案，評估其中 alpha 公式並輸出特徵欄位。
        2. inputs:
           - paths(Iterable[str|Path]): Alpha JSON 路徑（字串或 Path），可為單一路徑或列表。
        3. return: DataFrame，欄名格式為 'ALPHA__<檔名>'，dtype=float32。
        """
        if paths is None:
            raise ValueError("build_alpha 需要 paths 參數。")

        if isinstance(paths, (str, Path)):
            path_list = [paths]
        else:
            path_list = list(paths)

        if not path_list:
            raise ValueError("build_alpha 收到空的 paths。")

        features: Dict[str, pd.Series] = {}
        used_names: set[str] = set()

        for idx, raw_path in enumerate(path_list, start=1):
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(f"alpha 路徑不存在: {path}")

            try:
                alpha = load_alpha(path.as_posix())
            except Exception as exc:
                raise RuntimeError(f"載入 alpha 失敗: {path}") from exc

            try:
                values = alpha.tree.eval(self.df)
            except Exception as exc:
                raise RuntimeError(f"評估 alpha 失敗: {path}") from exc

            if isinstance(values, pd.DataFrame):
                if values.shape[1] != 1:
                    raise ValueError(f"alpha {path} 的輸出包含多個欄位，暫不支援。")
                values = values.iloc[:, 0]

            if not isinstance(values, pd.Series):
                values = pd.Series(values, index=self.df.index)

            if not values.index.equals(self.df.index):
                values = values.reindex(self.df.index)

            series = pd.to_numeric(values, errors="coerce").astype("float32")

            base = f"ALPHA__{path.stem or f'alpha_{idx}'}"
            base = re.sub(r"[^0-9A-Za-z_]+", "_", base)
            name = base
            suffix = 1
            while name in used_names:
                suffix += 1
                name = f"{base}__{suffix}"

            used_names.add(name)
            features[name] = series

        return pd.DataFrame(features, index=self.df.index)
    
    # === VectorBT（pandas_ta / TA-Lib） ===
    def build_VBT(self,
                all: bool = False,
                lib: str = "pandas_ta",
                items: list | None = None,
                include: list[str] | None = None,
                exclude: list[str] | None = None,
                per_params: dict | None = None,
                max_indicators: int | None = None) -> pd.DataFrame:
        """
        1. 說明: 以 vectorbt 一次產生大量技術指標。
                - all=True：掃描整個指標套件（pandas_ta / ta / talib），每個指標用預設參數跑，
                並可用 per_params 針對個別指標覆寫參數。
                - all=False：使用 items 列表（lib/fn/inputs/grid）逐項指定。
        2. inputs:
        - all(bool): 是否跑該套件所有可解析指標。
        - lib(str): 'pandas_ta' | 'ta' | 'talib'。
        - items(list|None): 非 all 模式時的指標規格。
        - include(list[str]|None): 名稱過濾（* ? [] 萬用字元）。
        - exclude(list[str]|None): 排除過濾。
        - per_params(dict|None): {'rsi': {'length':[7,14]}, 'bbands': {...}}（名稱大小寫不敏感）。
        - max_indicators(int|None): 上限，避免欄位爆炸。
        3. return:
        - DataFrame: 欄名格式 'short__out__param=...__...'（float32）。
        """
        import fnmatch
        import numpy as np
        import pandas as pd
        import vectorbt as vbt

        # ---- 輸入映射（依工廠 input_names 自動對應）----
        inputs_map = {
            "open":   self.ohlcv["open"],
            "high":   self.ohlcv["high"],
            "low":    self.ohlcv["low"],
            "close":  self.ohlcv["close"],
            "volume": self.ohlcv["volume"],
            # 常見別名
            "real":   self.ohlcv["close"],
            "input":  self.ohlcv["close"],
            "real0":  self.ohlcv["close"],
        }

        # ---- 取得並規一工廠清單 → List[(name, FactoryClass)] ----
        def _collect_factories(libname: str) -> list[tuple[str, type]]:
            if libname == "pandas_ta":
                raw = vbt.IndicatorFactory.get_pandas_ta_indicators(silence_warnings=True)
            elif libname == "ta":
                raw = vbt.IndicatorFactory.get_ta_indicators()
            elif libname == "talib":
                raw = vbt.IndicatorFactory.get_talib_indicators()
            else:
                raise ValueError(f"Unknown lib: {libname}")

            pairs: list[tuple[str, type]] = []
            if isinstance(raw, dict):
                pairs = [(str(k), v) for k, v in raw.items()]
            else:
                # set / list / tuple
                for item in list(raw):
                    F = item
                    if isinstance(item, str):
                        try:
                            if libname == "pandas_ta":
                                F = vbt.IndicatorFactory.from_pandas_ta(item)
                            elif libname == "ta":
                                F = vbt.IndicatorFactory.from_ta(item)
                            elif libname == "talib":
                                F = vbt.IndicatorFactory.from_talib(item)
                        except Exception:
                            continue
                        nm = str(item).lower()
                    else:
                        short_attr = getattr(F, "short_name", None)
                        if isinstance(short_attr, property):
                            short_attr = ""
                        nm = short_attr or getattr(F, "__name__", None) or "ind"
                        nm = str(nm).lower()
                    pairs.append((nm, F))

            # 去重名
            seen = {}
            uniq = []
            for nm, F in pairs:
                base = nm
                idx = 1
                while nm in seen:
                    idx += 1
                    nm = f"{base}_{idx}"
                seen[nm] = True
                uniq.append((nm, F))
            return uniq

        def _filter_names(names: list[str]) -> list[str]:
            keep = list(names)
            if include:
                inc = []
                for pat in include:
                    inc += [n for n in keep if fnmatch.fnmatch(n.lower(), pat.lower())]
                keep = sorted(set(inc))
            if exclude:
                for pat in exclude:
                    keep = [n for n in keep if not fnmatch.fnmatch(n.lower(), pat.lower())]
            return keep

        def _flatten_df(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
            if isinstance(df.columns, pd.MultiIndex):
                cols = ["__".join(map(str, c)) for c in df.columns]
            else:
                cols = [str(c) for c in df.columns]
            df = df.copy()
            df.columns = [f"{prefix}__{c}" for c in cols]
            return df

        def _run_factory(F: type, override: dict | None, short_hint: str) -> pd.DataFrame | None:
            # 準備輸入
            kw = {}
            for in_name in (F.input_names or []):
                key = str(in_name).lower()
                s = inputs_map.get(key, None)
                if s is None:
                    return None  # 缺必要輸入 → 跳過
                kw[in_name] = s

            # 參數覆寫（可單值或清單 → 自動形成參數網格）
            if override:
                kw.update(override)

            ind = F.run(**kw)

            parts = []
            short_attr = getattr(F, "short_name", None)
            if isinstance(short_attr, property):
                short_attr = ""
            short = short_hint or short_attr or getattr(F, "__name__", None) or "ind"
            short = str(short).lower()
            for out_name in (F.output_names or []):
                if not hasattr(ind, out_name):
                    continue
                df_out = getattr(ind, out_name)
                df_out = _flatten_df(pd.DataFrame(df_out), f"{short}__{out_name}")
                parts.append(df_out)
            if not parts:
                return None
            out = pd.concat(parts, axis=1)
            return out.replace([np.inf, -np.inf], np.nan).astype("float32")

        per_params = {str(k).lower(): v for k, v in (per_params or {}).items()}

        # ========= A) all 模式 =========
        if all:
            facs = _collect_factories(lib)                         # [(name, F)]
            names = _filter_names([n for n, _ in facs])
            if max_indicators is not None:
                names = names[:int(max_indicators)]
            names = list(names)
            name2F = {n: F for n, F in facs}
            outs, skipped = [], []
            iter_names = tqdm(names, desc=f"VBT[{lib}] indicators", leave=False) if names else names
            for name in iter_names:
                F = name2F[name]
                short_attr = getattr(F, "short_name", "")
                if isinstance(short_attr, property):
                    try:
                        short_attr = short_attr.fget(F)
                    except Exception:
                        short_attr = ""
                short_key = str(short_attr).lower() if short_attr else ""
                over = per_params.get(name) or (short_key and per_params.get(short_key))
                try:
                    df_one = _run_factory(F, over, name)
                except Exception:
                    df_one = None
                if df_one is None:
                    skipped.append(name)
                else:
                    outs.append(df_one)
            if not outs:
                return pd.DataFrame(index=self.df.index)
            return pd.concat(outs, axis=1)

        # ========= B) items 模式（相容你現有配置） =========
        if not items:
            return pd.DataFrame(index=self.df.index)

        outs = []
        iter_items = tqdm(items, desc=f"VBT[{lib}] items", leave=False) if items else items
        for it in iter_items:
            lib_i = str(it.get("lib", lib)).lower()
            fn = str(it["fn"]).lower()
            inputs_spec = it.get("inputs", "close")
            grid = dict(it.get("grid", {}))

            # 建工廠
            if lib_i == "pandas_ta":
                F = vbt.IndicatorFactory.from_pandas_ta(fn)
            elif lib_i == "ta":
                F = vbt.IndicatorFactory.from_ta(fn)
            elif lib_i == "talib":
                F = vbt.IndicatorFactory.from_talib(fn)
            else:
                continue

            # 對應輸入（支援 'close' 或 ['high','low','close']）
            kw = {}
            in_names = list(F.input_names or [])
            if isinstance(inputs_spec, (list, tuple)):
                for in_name, src_name in zip(in_names, inputs_spec):
                    s = inputs_map.get(str(src_name).lower(), None)
                    if s is not None:
                        kw[in_name] = s
            else:
                # 單一字串：盡量匹配所有 input_names
                src = inputs_map.get(str(inputs_spec).lower(), None)
                for in_name in in_names:
                    if src is not None and in_name not in kw:
                        kw[in_name] = src

            kw.update(grid)  # 參數網格

            ind = F.run(**kw)
            short = (getattr(F, "short_name", None) or fn).lower()
            for out_name in (F.output_names or []):
                if not hasattr(ind, out_name):
                    continue
                df_out = getattr(ind, out_name)
                df_out = _flatten_df(pd.DataFrame(df_out), f"{short}__{out_name}")
                outs.append(df_out)

        if not outs:
            return pd.DataFrame(index=self.df.index)
        return pd.concat(outs, axis=1).replace([np.inf, -np.inf], np.nan).astype("float32")

    # === Kats TsFeatures（滑窗抽取） ===
    def build_TSF(self,
                targets: List[Dict[str, Any]],
                tsf_params: Dict[str, Any] | None = None) -> pd.DataFrame:
        """
        1. 說明: 使用 Nixtla 的 tsfeatures 套件，對多個單變量序列在多個滑動視窗上抽取
                經典時間序列特徵（趨勢/季節性/ACF/熵/flat_spots 等），並與索引對齊輸出寬表。
        2. inputs:
        - targets (List[dict]): 每個 dict 形如
                {
                "name": "<欄位名，如 'close' 或 'volume'>",
                "transform": "raw" | "logret" | "pct",
                "windows": ["6h","1d", ...]  # 對每個時間點，使用 (t - window, t] 的資料段
                }
        - tsf_params (dict|None): 轉交 tsfeatures() 的參數（常用 'freq': int 季節長度）。
                                    例如 15m K、以日季節 → {"freq": 96}
        3. return: DataFrame（float32），欄名格式：
                TSF__<name>__<transform>__win=<w>__<feature_name>
                注意：此函式不做 shift，交由 FeatureComputer 統一 shift(1) 防外洩。
        """

        params = dict(tsf_params or {})

        def _series(col: str, transform: str) -> pd.Series:
            s = pd.to_numeric(self.df[col], errors="coerce").astype("float32")
            if transform == "logret":
                s = np.log(s).diff()
            elif transform == "pct":
                s = s.pct_change()
            return s

        def _panel_from_seg(seg: pd.Series) -> pd.DataFrame:
            # tsfeatures 需要 panel：unique_id, ds, y；ds 建議用 naive datetime
            idx = pd.DatetimeIndex(seg.index)
            if idx.tz is not None:
                ds = idx.tz_convert("UTC").tz_localize(None)
            else:
                ds = idx
            return pd.DataFrame({
                "unique_id": "S",  # 單序列
                "ds": ds,
                "y": seg.values
            })

        parts: List[pd.DataFrame] = []
        targets_list = list(targets)
        iter_targets = tqdm(targets_list, desc="TSF targets", leave=False) if targets_list else targets_list
        for tgt in iter_targets:
            name = tgt["name"]
            tfm  = str(tgt.get("transform", "raw"))
            wins = list(tgt.get("windows", []))
            y = _series(name, tfm)

            iter_wins = tqdm(wins, desc=f"TSF[{name}:{tfm}] windows", leave=False) if wins else wins
            for w in iter_wins:
                rows = []
                iter_idx = tqdm(y.index, desc=f"TSF[{name}:{tfm}] win={w}", leave=False) if len(y.index) else y.index
                for end in iter_idx:
                    start = end - pd.Timedelta(w)
                    seg = y.loc[(y.index > start) & (y.index <= end)].dropna()
                    if len(seg) < 8:           # 視窗太短跳過，以避免不穩定估計
                        rows.append({})
                        continue
                    panel = _panel_from_seg(seg)
                    # 只傳 tsfeatures 認得的參數（目前最重要的是 freq）
                    feats_df = _tsf(panel, **{k: v for k, v in params.items() if k in ("freq",)})
                    # tsfeatures 回傳一列（每個 unique_id 一列）
                    feats = feats_df.iloc[0].to_dict()
                    rows.append(feats)

                df_feat = pd.DataFrame(rows, index=y.index)
                df_feat = df_feat.add_prefix(f"TSF__{name}__{tfm}__win={w}__")
                parts.append(df_feat)

        out = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).astype("float32")
        return out
    
    # ---------- Name → Builder ----------
    @property
    def builders(self) -> Dict[str, callable]:
        """
        1. 說明: 提供 '指標名稱' → '建構函式' 的映射，對外統一呼叫介面。
        2. inputs: 無。
        3. return: Dict[str, callable]，呼叫方式為 builders[name](kwargs)。
        """
        return {
            # 原始（特徵用：一律在 FeatureComputer 統一 shift(1)；名稱標示 *_L1）
            "OPEN":   lambda kw: self.df[["open"]].rename(columns={"open": "OPEN_L1"}),
            "HIGH":   lambda kw: self.df[["high"]].rename(columns={"high": "HIGH_L1"}),
            "LOW":    lambda kw: self.df[["low"]].rename(columns={"low": "LOW_L1"}),
            "CLOSE":  lambda kw: self.df[["close"]].rename(columns={"close": "CLOSE_L1"}),
            "VOLUME": lambda kw: self.df[["volume"]].rename(columns={"volume": "VOLUME_L1"}),

            # Trend
            "SMA":    lambda kw: self.build_SMA(**kw),
            "EMA":    lambda kw: self.build_EMA(**kw),
            "TEMA":   lambda kw: self.build_TEMA(**kw),
            "MACD":   lambda kw: self.build_MACD(**kw),
            "SLOPE":  lambda kw: self.build_SLOPE(**kw),
            "TTM_TRND": lambda kw: self.build_TTM_TRND(**kw),
            "DPO":    lambda kw: self.build_DPO(**kw),
            "AMATE_LR": lambda kw: self.build_AMATE_LR(**kw),

            # Momentum / Oscillator
            "RSI":    lambda kw: self.build_RSI(**kw),
            "MOM":    lambda kw: self.build_MOM(**kw),
            "STOCH":  lambda kw: self.build_STOCH(**kw),
            "KDJ":    lambda kw: self.build_KDJ(**kw),
            "UO":     lambda kw: self.build_UO(**kw),
            "RVI":    lambda kw: self.build_RVI(**kw),
            "CCI":    lambda kw: self.build_CCI(**kw),
            "ZS":     lambda kw: self.build_ZS(**kw),
            "WILLR":  lambda kw: self.build_WILLR(**kw),

            # Volatility
            "TRUERANGE": lambda kw: self.build_TRUERANGE(),
            "RANGE":     lambda kw: self.build_RANGE(**kw),
            "ATR":       lambda kw: self.build_ATR(**kw),
            "MASSI":     lambda kw: self.build_MASSI(**kw),
            "BBP":       lambda kw: self.build_BBP(**kw),
            "EWMRET":    lambda kw: self.build_EWMRET(**kw),

            # Price-Volume
            "PVO":     lambda kw: self.build_PVO(**kw),
            "PVR":     lambda kw: self.build_PVR(),
            "BOP":     lambda kw: self.build_BOP(),
            "PXVOL":   lambda kw: self.build_PXVOL(),

            # Returns / Time
            "LOGRET":  lambda kw: self.build_LOGRET(**kw),
            "TIME_CYC": lambda kw: self.build_TIME_CYC(**kw),

            # funding rate
            "FOUND":    lambda kw: self.df[["funding_rate"]],

            # 15-min info (for 1H 尺度ohlvc，15min不用)
            "M15_DIR": lambda kw: self.build_15_DIR(),
            "M15_VOL": lambda kw: self.build_15_VOL(),

            # FNG info
            "FNG_IDX": lambda kw: self.build_fng(),
            "ALPHA": lambda kw: self.build_alpha(**kw),

            "VBT":  lambda kw: self.build_VBT(**kw),     # vectorbt + (pandas_ta / TA-Lib)
            "TSF": lambda kw: self.build_TSF(**kw),
        }
