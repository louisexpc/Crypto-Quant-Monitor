# features/indicators.py
from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import Dict, Any, Union, Iterable, List
import re
import numpy as np
import pandas as pd
import pandas_ta as ta

DEFAULT_MINUTE_PREFIXES = ("m_",)
# _M_PAT = re.compile(r"^m_-(\d+)_(.+)$")  # 解析 m_-{lag}_{base}
_M_PAT = re.compile(r"^m_(-?\d+)_(.+)$")

# =========================
# 1) 指標庫：只負責把 self.df → 各種特徵
# =========================
class IndicatorLibrary:
    """
    - 規格化 df 為 UTC DatetimeIndex + 只留 OHLCV(float32)
    - builder 皆為 instance method（用 self.df/self.ohlcv）
    - 對外提供 self.builders（name → callable(kwargs)）
    """

    def __init__(self,
                df_raw: pd.DataFrame,
                *,
                freq_check: str | None = None,
                prefer_time_col: str = "timestamp"):
                self.freq_check = freq_check
                self.prefer_time_col = prefer_time_col
                self.df = self._normalize_ohlcv(df_raw)
                self.ohlcv = {
                    "open":   self.df["open"],
                    "high":   self.df["high"],
                    "low":    self.df["low"],
                    "close":  self.df["close"],
                    "volume": self.df["volume"],
                }

    # ---------- 基礎規格化 ----------
    def _normalize_ohlcv(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        df.columns = [c.lower().strip() for c in df.columns]

        # 1) 建立 UTC 索引（優先用 prefer_time_col，其次 timestamp/datetime）
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
            if self.prefer_time_col in df.columns:
                idx = self._make_utc_index_from_col(df[self.prefer_time_col], self.prefer_time_col)
            elif "timestamp" in df.columns:
                idx = self._make_utc_index_from_col(df["timestamp"], "timestamp")
            elif "datetime" in df.columns:
                idx = self._make_utc_index_from_col(df["datetime"], "datetime")
            else:
                raise ValueError("需要 DatetimeIndex 或 'timestamp'/'datetime' 欄位")
            df.index = idx

        # 2) 排序、去重（保留最後一筆）
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # 3) 確認 OHLCV 存在；只把 OHLCV 轉成 float32，其他欄位一律保留
        need = ["open", "high", "low", "close", "volume"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"缺少欄位: {miss}")
        for c in need:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

        # 4) （可選）頻率檢查：只警告，不補、不砍
        if self.freq_check and len(df) > 1:
            expected = pd.date_range(df.index[0], df.index[-1], freq=self.freq_check, tz="UTC")
            miss_idx = expected.difference(df.index)
            if len(miss_idx) > 0:
                print(f"[WARN] 缺少 {len(miss_idx)} 根 {self.freq_check} K；預設不補齊。")

        return df

    @staticmethod
    def _make_utc_index_from_col(col: pd.Series, kind: str) -> pd.DatetimeIndex:
        if kind == "timestamp":
            ts = col.astype("int64")
            unit = "ms" if ts.iloc[0] > 1_000_000_000_000 else "s"
            return pd.to_datetime(ts, unit=unit, utc=True)
        elif kind == "datetime":
            return pd.to_datetime(col, utc=True)
        else:
            raise ValueError(f"未知時間欄位型態：{kind}")

    # ---------- sign 參數解析：True/False/'both' ----------
    @staticmethod
    def _resolve_sign_mode(sign) -> str:
        if isinstance(sign, str):
            s = sign.strip().lower()
            if s in ("both",): return "both"
            if s in ("true","y","1","sign"): return "sign"
            if s in ("false","n","0","cont","continuous"): return "cont"
            raise ValueError(f"Invalid sign param: {sign}")
        return "sign" if sign is True else "cont"

    # ========== Trend / Averages ==========
    def build_SMA(self, length: int, sign=False) -> pd.DataFrame:
        mode = self._resolve_sign_mode(sign)
        sma = ta.sma(self.ohlcv["close"], length=length)
        out = {}
        if mode in ("cont","both"):
            out[f"SMA_{length}"] = sma.astype(np.float32)
        if mode in ("sign","both"):
            sig = np.where(self.ohlcv["close"].values > sma.values, 1.0,
                           np.where(self.ohlcv["close"].values < sma.values, -1.0, 0.0))
            out[f"SSMA_{length}"] = pd.Series(sig, index=self.df.index, dtype=np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_EMA(self, length: int, sign=False) -> pd.DataFrame:
        mode = self._resolve_sign_mode(sign)
        ema = ta.ema(self.ohlcv["close"], length=length)
        out = {}
        if mode in ("cont","both"):
            out[f"EMA_{length}"] = ema.astype(np.float32)
        if mode in ("sign","both"):
            sig = np.where(self.ohlcv["close"].values > ema.values, 1.0,
                           np.where(self.ohlcv["close"].values < ema.values, -1.0, 0.0))
            out[f"SEMA_{length}"] = pd.Series(sig, index=self.df.index, dtype=np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_TEMA(self, length: int, sign=False) -> pd.DataFrame:
        mode = self._resolve_sign_mode(sign)
        tema = ta.tema(self.ohlcv["close"], length=length)
        out = {}
        if mode in ("cont","both"):
            out[f"TEMA_{length}"] = tema.astype(np.float32)
        if mode in ("sign","both"):
            sig = np.where(self.ohlcv["close"].values > tema.values, 1.0,
                           np.where(self.ohlcv["close"].values < tema.values, -1.0, 0.0))
            out[f"STEMA_{length}"] = pd.Series(sig, index=self.df.index, dtype=np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_MACD(self, fast: int = 12, slow: int = 26, signal: int = 9, sign=False) -> pd.DataFrame:
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

    def build_SLOPE(self, length: int) -> pd.DataFrame:
        s = ta.slope(self.ohlcv["close"], length=length)
        return pd.DataFrame({f"SLOPE_{length}": s.astype(np.float32)}, index=self.df.index)
    
    def build_TTM_TRND(self, length: int = 6) -> pd.DataFrame:
        tt = ta.ttm_trend(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], length=length)
        s  = (tt.iloc[:, 0] if isinstance(tt, pd.DataFrame) else tt)          # 只取第一欄；若是 Series 直接用
        s  = pd.Series(s, index=getattr(s, "index", self.df.index)).reindex(self.df.index).astype(np.float32)
        return s.rename(f"TTM_TRND_{length}").to_frame()

    def build_DPO(self, length: int) -> pd.DataFrame:
        s = ta.dpo(self.ohlcv["close"], length=length, centered=False)
        return pd.DataFrame({f"DPO_{length}": s.astype(np.float32)}, index=self.df.index)

    def build_AMATE_LR(self, fast: int, slow: int, mamode: int = 2) -> pd.DataFrame:
        amat = ta.amat(self.ohlcv["close"], fast=fast, slow=slow, mamode=mamode)
        col = [c for c in amat.columns if "AMATe_LR" in c or "LR_" in c]
        tgt = col[0] if col else amat.columns[0]
        return pd.DataFrame({f"AMATe_LR_{fast}_{slow}_{mamode}": amat[tgt].astype(np.float32)}, index=self.df.index)

    # ========== Momentum / Oscillator ==========
    def build_RSI(self, length: int) -> pd.DataFrame:
        out = ta.rsi(self.ohlcv["close"], length=length)
        return pd.DataFrame({f"RSI_{length}": out.astype(np.float32)}, index=self.df.index)

    def build_MOM(self, length: int) -> pd.DataFrame:
        out = ta.mom(self.ohlcv["close"], length=length)
        return pd.DataFrame({f"MOM_{length}": out.astype(np.float32)}, index=self.df.index)

    def build_STOCH(self, k: int, d: int = 3, smooth_k: int = 3) -> pd.DataFrame:
        st = ta.stoch(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], k=k, d=d, smooth_k=smooth_k)
        st = st.rename(columns={st.columns[0]: f"STOCHk_{k}", st.columns[1]: f"STOCHd_{k}"})
        return st[[f"STOCHk_{k}", f"STOCHd_{k}"]].astype(np.float32)

    def build_KDJ(self, k: int, d: int, smooth_k: int) -> pd.DataFrame:
        st = ta.stoch(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], k=k, d=d, smooth_k=smooth_k)
        kcol, dcol = st.columns[0], st.columns[1]
        j = 3 * st[kcol] - 2 * st[dcol]
        return pd.DataFrame({f"J_{k}_{d}": j.astype(np.float32)}, index=self.df.index)

    def build_UO(self, fast: int, medium: int, slow: int) -> pd.DataFrame:
        s = ta.uo(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], fast=fast, medium=medium, slow=slow)
        return pd.DataFrame({f"UO_{fast}_{medium}_{slow}": s.astype(np.float32)}, index=self.df.index)

    def build_RVI(self, length: int) -> pd.DataFrame:
        rvi = ta.rvi(self.ohlcv["close"], length=length)
        if isinstance(rvi, pd.DataFrame):
            col = [c for c in rvi.columns if c.startswith("RVI")]
            tgt = col[0] if col else rvi.columns[0]
            s = rvi[tgt]
        else:
            s = rvi
        return pd.DataFrame({f"RVI_{length}": s.astype(np.float32)}, index=self.df.index)

    def build_CCI(self, length: int, c: float = 0.015) -> pd.DataFrame:
        s = ta.cci(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], length=length, c=c)
        return pd.DataFrame({f"CCI_{length}_{c}": s.astype(np.float32)}, index=self.df.index)

    def build_ZS(self, length: int) -> pd.DataFrame:
        s = ta.zscore(self.ohlcv["close"], length=length)
        return pd.DataFrame({f"ZS_{length}": s.astype(np.float32)}, index=self.df.index)

    def build_WILLR(self, length: int) -> pd.DataFrame:
        s = ta.willr(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], length=length)
        return pd.DataFrame({f"WILLR_{length}": s.astype(np.float32)}, index=self.df.index)

    # ========== Volatility ==========
    def build_TRUERANGE(self) -> pd.DataFrame:
        prev_close = self.ohlcv["close"].shift(1)
        tr = pd.concat([
            (self.ohlcv["high"] - self.ohlcv["low"]),
            (self.ohlcv["high"] - prev_close).abs(),
            (self.ohlcv["low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        return pd.DataFrame({"TRUERANGE_1": tr.astype(np.float32)}, index=self.df.index)

    def build_RANGE(self, window: int, pct: bool = True) -> pd.DataFrame:
        H, L, C = self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"]
        hl = (H - L)
        if pct:
            base = C.abs().replace(0, np.nan)
            s = (hl / base).rolling(window, min_periods=max(1, window//2)).mean()
        else:
            s = hl.rolling(window, min_periods=max(1, window//2)).mean()
        return pd.DataFrame({f"RANGE_{window}": s.astype(np.float32)}, index=self.df.index)

    def build_ATR(self, length: int = 14, pct: bool = True) -> pd.DataFrame:
        atr = ta.atr(self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"], length=length)
        out = {f"ATR_{length}": atr.astype(np.float32)}
        if pct:
            C = self.ohlcv["close"].abs().replace(0, np.nan)
            out[f"ATRP_{length}"] = (atr / C).astype(np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def build_MASSI(self, fast: int, slow: int) -> pd.DataFrame:
        s = ta.massi(self.ohlcv["high"], self.ohlcv["low"], fast=fast, slow=slow)
        return pd.DataFrame({f"MASSI_{fast}_{slow}": s.astype(np.float32)}, index=self.df.index)

    def build_BBP(self, length: int, std: float) -> pd.DataFrame:
        bb = ta.bbands(self.ohlcv["close"], length=length, std=std)
        col = [c for c in bb.columns if c.startswith("BBP_")]
        tgt = col[0] if col else bb.columns[-1]
        return pd.DataFrame({f"BBP_{length}_{std}": bb[tgt].astype(np.float32)}, index=self.df.index)

    def build_EWMRET(self, halflife: int | Iterable[int]) -> pd.DataFrame:
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
        pvo = ta.pvo(self.ohlcv["volume"], fast=fast, slow=slow, signal=signal)
        col = [c for c in pvo.columns if c.startswith("PVO_") and c.count("_") >= 2]
        main = col[0] if col else pvo.columns[0]
        return pd.DataFrame({main: pvo[main].astype(np.float32)}, index=self.df.index)

    def build_PVR(self) -> pd.DataFrame:
        s = ta.pvr(self.ohlcv["close"], self.ohlcv["volume"])
        if isinstance(s, pd.Series):
            return pd.DataFrame({"PVR": s.astype(np.float32)}, index=self.df.index)
        return pd.DataFrame({s.columns[0] if len(s.columns) else "PVR": s.iloc[:,0].astype(np.float32)}, index=self.df.index)

    def build_BOP(self) -> pd.DataFrame:
        s = ta.bop(self.ohlcv["open"], self.ohlcv["high"], self.ohlcv["low"], self.ohlcv["close"])
        return pd.DataFrame({"BOP": s.astype(np.float32)}, index=self.df.index)

    def build_PXVOL(self) -> pd.DataFrame:
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
        以 m15_{0..3}_close 產出三段 1/0 漲跌指標：
        M15_DIR_01: (m15_1_close > m15_0_close)
        M15_DIR_12: (m15_2_close > m15_1_close)
        M15_DIR_23: (m15_3_close > m15_2_close)
        """
        x = self.df  # 含 m15_* 的完整表
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
        直接輸出四段 15m 量能（原樣），欄位：
        M15_VOL_0/1/2/3 對應 m15_0/1/2/3 的 vol
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
        直接從 self.df 讀取已併入的 FNG 欄位，回傳三欄（不在此處 shift）：
        - sent_fng, sent_fng_diff1, sent_fng_z7d
        若缺欄位會丟出清楚的錯誤訊息。
        """
        x = self.df
        need = ["sent_fng", "sent_fng_diff1", "sent_fng_z7d"]
        missing = [c for c in need if c not in x.columns]
        if missing:
            raise ValueError(
                f"缺少 FNG 欄位：{missing}。請先把 fng_15m_utc.csv 併入主表 "
                f"(或用我給的 merge_fng_into_15m.py) 再呼叫 build_fng。"
            )

        out = pd.DataFrame({
            "sent_fng":        pd.to_numeric(x["sent_fng"],        errors="coerce").astype("float32"),
            "sent_fng_diff1":  pd.to_numeric(x["sent_fng_diff1"],  errors="coerce").astype("float32"),
            "sent_fng_z7d":    pd.to_numeric(x["sent_fng_z7d"],    errors="coerce").astype("float32"),
        }, index=self.df.index)

        return out



    # ---------- Name → Builder ----------
    @property
    def builders(self) -> Dict[str, callable]:
        return {
            # 原始
            "OPEN":   lambda kw: self.df[["open"]],
            "HIGH":   lambda kw: self.df[["high"]],
            "LOW":    lambda kw: self.df[["low"]],
            "CLOSE":  lambda kw: self.df[["close"]],
            "VOLUME": lambda kw: self.df[["volume"]],

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
            "M15_DIR": lambda kw: self.build_15_DIR(),  # 三段 1/0
            "M15_VOL": lambda kw: self.build_15_VOL(),  # 四段量能

            # FNG info
            "FNG_IDX": lambda kw: self.build_fng()
        }


# =========================
# 2) 特徵計算器：features → 計算 + 快取 + manifest
# =========================
class FeatureComputer:
    def __init__(self, lib: IndicatorLibrary, cache_dir: Union[str, Path] = "cache_features"):
        self.lib = lib
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --- 指紋 & 路徑 ---
    @staticmethod
    def _fingerprint_df(df_ohlcv: pd.DataFrame) -> str:
        sig = pd.Series({
            "rows": int(len(df_ohlcv)),
            "idx0": str(df_ohlcv.index[0]) if len(df_ohlcv) else "NA",
            "idxN": str(df_ohlcv.index[-1]) if len(df_ohlcv) else "NA",
            "sum_close": float(np.nan_to_num(df_ohlcv["close"].sum())),
            "sum_vol": float(np.nan_to_num(df_ohlcv["volume"].sum())),
        })
        return hashlib.md5(sig.to_json().encode()).hexdigest()[:10]

    @staticmethod
    def _fingerprint_plan(plan: Dict[str, Any]) -> str:
        payload = json.dumps(plan, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:10]

    def _cache_path(self, df_fp: str, plan_fp: str) -> Path:
        return self.cache_dir / f"feat_{df_fp}_{plan_fp}.parquet"

    def _manifest_path(self, df_fp: str, plan_fp: str) -> Path:
        return self.cache_dir / f"feat_{df_fp}_{plan_fp}.manifest.json"

    # --- 規格鍵（name+kwargs → 穩定 key）---
    @staticmethod
    def _spec_key(name: str, kwargs: dict) -> str:
        def _val2str(v):
            if isinstance(v, (list, tuple)):
                return "[" + ",".join(str(x) for x in v) + "]"
            if isinstance(v, bool):
                return "true" if v else "false"
            return str(v)
        items = sorted((k, _val2str(v)) for k, v in (kwargs or {}).items())
        return f"{name.upper()}({','.join(f'{k}={v}' for k,v in items)})" if items else name.upper()

    # --- 只取 features ---
    def _prune_features_only(self, plan: dict) -> list[dict]:
        feats = [f for f in (plan.get("features") or []) if f.get("enabled", False)]
        return feats

    # --- compute（含快取+manifest；不使用 groups）---
    def compute(self, plan: Dict[str, Any], cfg, *, load_if_exists: bool = True) -> pd.DataFrame:
        """
        計算（或從快取載入）特徵欄位。

        - 預設行為：若對應 parquet 快取存在且 `load_if_exists=True`，直接讀取；否則重算並寫入 parquet 與 manifest。
        - dropna 時機：可由 cfg["features"].get("dropna", True) 控制；True 表示立刻刪除含 NaN 的列。
        """
        feat_list = self._prune_features_only(plan)
        if not feat_list:
            raise ValueError("計畫沒有任何 enabled=True 的 features。")

        # 保留1-min欄位
        keep_prefixes = DEFAULT_MINUTE_PREFIXES

        eff_plan = {
            "features": [{"name": f["name"], "kwargs": f.get("kwargs", {})} for f in feat_list],
            "keep_prefixes": list(keep_prefixes),
        }

        df_fp = self._fingerprint_df(self.lib.df)
        pl_fp = self._fingerprint_plan(eff_plan)
        cpath = self._cache_path(df_fp, pl_fp)
        mpath = self._manifest_path(df_fp, pl_fp)

        # 若有快取，優先讀取
        if load_if_exists and cpath.exists():
            X = pd.read_parquet(cpath)
            # 仍做數值清理，避免舊快取中殘留 inf
            X = X.replace([np.inf, -np.inf], np.nan)
            if (cfg.get("features", {}) or {}).get("dropna", True):
                X = X.dropna()
            return X.astype("float32")

        # 計算所有 feature parts
        parts, manifest = [], {}
        for item in feat_list:
            name = str(item["name"]).upper()
            kwargs = item.get("kwargs", {}) or {}
            if name not in self.lib.builders:
                raise ValueError(f"Unknown indicator: {name}")
            feat = self.lib.builders[name](kwargs)
            feat = feat.shift(1)  # 防洩漏：統一 shift(1)

            spec = self._spec_key(name, kwargs)
            manifest[spec] = list(map(str, feat.columns))
            parts.append(feat)

        # 保留 minute 前綴欄位（不看 enabled）
        prefixes = tuple(keep_prefixes)
        min_feat = list(cfg.get("features", {}).get("min_trade_feat", []))

        passthrough_cols = []
        for c in self.lib.df.columns:
            s = str(c)
            if not s.startswith(prefixes):
                continue
            m = _M_PAT.match(s)
            if not m:
                continue
            base = m.group(2)
            if base in min_feat:
                passthrough_cols.append(s)

        if passthrough_cols:
            mdf = self.lib.df.loc[:, passthrough_cols].copy()
            mdf = mdf.shift(1)
            for c in mdf.columns:
                mdf[c] = pd.to_numeric(mdf[c], errors="coerce").astype("float32")
            for p in prefixes:
                keep_cols = [c for c in passthrough_cols if str(c).startswith(p)]
                if keep_cols:
                    manifest[f"_PASSTHROUGH({p})"] = list(map(str, keep_cols))
            parts.append(mdf)

        # 整併與檢查重複
        X = pd.concat(parts, axis=1).astype("float32")
        X = X.replace([np.inf, -np.inf], np.nan)
        if (cfg.get("features", {}) or {}).get("dropna", True):
            X = X.dropna()

        dups = X.columns[X.columns.duplicated()]
        if len(dups):
            raise ValueError(f"Duplicate feature names: {list(dups)}")

        # 寫入快取與 manifest
        X.to_parquet(cpath, index=True)
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return X

    # --- 欄位查詢（依當前 features 規格；不使用 groups）---
    def _load_manifest(self, plan: Dict[str, Any], cfg) -> Dict[str, list[str]]:
        feat_list = self._prune_features_only(plan)
        eff_plan = {
            "features": [{"name": f["name"], "kwargs": f.get("kwargs", {})} for f in feat_list],
            "keep_prefixes": list(DEFAULT_MINUTE_PREFIXES),  # 與 compute() 一致
        }
        df_fp = self._fingerprint_df(self.lib.df)
        pl_fp = self._fingerprint_plan(eff_plan)
        mpath = self._manifest_path(df_fp, pl_fp)
        if not mpath.exists():
            _ = self.compute(plan, cfg)
        with open(mpath, "r", encoding="utf-8") as f:
            return json.load(f)

    def columns_for_plan(self, plan: Dict[str, Any], cfg=None) -> List[str]:
        """
        回傳此 plan 對應的所有特徵欄位名稱（含 minute 直通欄位）。

        - 先依 features 規格（name+kwargs）收集欄位
        - 再附加 manifest 中的 _PASSTHROUGH(...) 條目（例如 m_ 前綴的 1-min 打平欄位）
        可透過 cfg["features"].get("include_passthrough", True) 控制是否包含直通欄位。
        """
        feat_list = self._prune_features_only(plan)
        mani = self._load_manifest(plan, cfg)
        cols = []
        for item in feat_list:
            spec = self._spec_key(str(item["name"]).upper(), item.get("kwargs", {}) or {})
            cols.extend(mani.get(spec, []))

        include_passthrough = True
        if cfg is not None:
            include_passthrough = bool((cfg.get("features", {}) or {}).get("include_passthrough", True))
        if include_passthrough:
            for k, v in mani.items():
                if isinstance(k, str) and k.startswith("_PASSTHROUGH("):
                    cols.extend(list(v))

        # 去重並保持順序
        seen = set()
        out = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def columns_for_feature(self, name: str, kwargs: dict | None = None,
                            plan: Dict[str, Any] | None = None, cfg=None) -> List[str]:
        if plan is None:
            tmp = {"features": [{"name": name, "kwargs": kwargs or {}, "enabled": True}]}
            return self.columns_for_plan(tmp,cfg)
        mani = self._load_manifest(plan,cfg)
        return mani.get(self._spec_key(str(name).upper(), kwargs or {}), [])
