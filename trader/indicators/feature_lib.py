from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as pta


class FeatureLibPTA:
    """PTA-only feature library for OHLCV-based feature computation.

    Args:
        df: Input dataframe that must contain open/high/low/close/volume/fng.
    Returns:
        None
    """

    _RAW_COLUMNS = {"open", "high", "low", "close", "volume", "fng"}

    def __init__(self, df: pd.DataFrame):
        """Initialize feature library with typed OHLCV views.

        Args:
            df: Input dataframe containing market columns and fng.
        Returns:
            None
        """
        need = ["open", "high", "low", "close", "volume", "fng"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"FeatureLibPTA missing required columns: {missing}")

        self.df = df
        self.o = pd.to_numeric(df["open"], errors="coerce").astype(float)
        self.h = pd.to_numeric(df["high"], errors="coerce").astype(float)
        self.l = pd.to_numeric(df["low"], errors="coerce").astype(float)
        self.c = pd.to_numeric(df["close"], errors="coerce").astype(float)
        self.v = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype(float)
        self.g = pd.to_numeric(df["fng"], errors="coerce").astype(float)

    @staticmethod
    def _nan_series(index: pd.Index) -> pd.Series:
        """Create a NaN-filled series aligned to index.

        Args:
            index: Target index.
        Returns:
            NaN series.
        """
        return pd.Series(np.nan, index=index, dtype="float64")

    def _as_series(self, out: object, *, include: Sequence[str] | None = None, exclude: Sequence[str] = ()) -> pd.Series:
        """Convert PTA output into a single series with optional column filtering.

        Args:
            out: PTA output object.
            include: Keywords that target column names.
            exclude: Keywords to exclude from target column names.
        Returns:
            Single feature series aligned to self.df index.
        """
        if out is None:
            return self._nan_series(self.df.index)
        if isinstance(out, pd.Series):
            return out.reindex(self.df.index)
        if not isinstance(out, pd.DataFrame) or out.shape[1] == 0:
            return self._nan_series(self.df.index)

        cols = [str(c) for c in out.columns]
        cols_l = [c.lower() for c in cols]
        include = tuple((include or ()))
        include_l = tuple(x.lower() for x in include)
        exclude_l = tuple(x.lower() for x in exclude)

        if include_l:
            for i, c in enumerate(cols_l):
                has_inc = any(k in c for k in include_l)
                has_exc = any(k in c for k in exclude_l)
                if has_inc and not has_exc:
                    return out.iloc[:, i].reindex(self.df.index)

        return out.iloc[:, 0].reindex(self.df.index)

    @staticmethod
    def _sanitize_feature(
        s: pd.Series,
        *,
        fillna_value: float | None = 0.0,
        dtype: str | None = "float32",
        shift: int = 0,
    ) -> pd.Series:
        """Sanitize feature values with optional shift/fill/cast.

        Args:
            s: Raw feature series.
            fillna_value: Fill value for NaN. Use None to skip filling.
            dtype: Output dtype. Use None to skip casting.
            shift: Number of bars to shift.
        Returns:
            Sanitized feature series.
        """
        out = pd.to_numeric(s, errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan)
        if int(shift) != 0:
            out = out.shift(int(shift))
        if fillna_value is not None:
            out = out.fillna(float(fillna_value))
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def _sanitize_frame(
        self,
        df: pd.DataFrame,
        *,
        fillna_value: float | None = 0.0,
        dtype: str | None = "float32",
        shift: int = 0,
    ) -> pd.DataFrame:
        """Sanitize each column in dataframe using unified series sanitizer.

        Args:
            df: Raw feature dataframe.
            fillna_value: Fill value for NaN. Use None to skip filling.
            dtype: Output dtype. Use None to skip casting.
            shift: Number of bars to shift.
        Returns:
            Sanitized dataframe.
        """
        cleaned = {}
        for col in df.columns:
            cleaned[str(col)] = self._sanitize_feature(
                pd.Series(df[col], index=self.df.index),
                fillna_value=fillna_value,
                dtype=dtype,
                shift=shift,
            )
        return pd.DataFrame(cleaned, index=self.df.index)

    def add_feat(
        self,
        fn: Any,
        lengths: Iterable[int],
        *,
        fillna_value: float | None = 0.0,
        dtype: str | None = "float32",
        shift: int = 0,
    ) -> None:
        """Generate multi-length features from a length-based method.

        Args:
            fn: Callable(length)->Series.
            lengths: Window lengths.
            fillna_value: Fill value for NaN. Use None to skip filling.
            dtype: Output dtype. Use None to skip casting.
            shift: Number of bars to shift.
        Returns:
            None
        """
        name = str(fn.__name__)
        for length in lengths:
            s = fn(int(length))
            self.df[f"{name}_{int(length)}"] = self._sanitize_feature(
                s,
                fillna_value=fillna_value,
                dtype=dtype,
                shift=shift,
            )

    def add_series(
        self,
        name: str,
        s: pd.Series,
        *,
        fillna_value: float | None = 0.0,
        dtype: str | None = "float32",
        shift: int = 0,
    ) -> None:
        """Write one feature series into dataframe with unified sanitation.

        Args:
            name: Feature name.
            s: Input series.
            fillna_value: Fill value for NaN. Use None to skip filling.
            dtype: Output dtype. Use None to skip casting.
            shift: Number of bars to shift.
        Returns:
            None
        """
        self.df[str(name)] = self._sanitize_feature(
            s,
            fillna_value=fillna_value,
            dtype=dtype,
            shift=shift,
        )

    def open(self) -> pd.Series:
        """Return open series.

        Args:
            None
        Returns:
            Open series.
        """
        return self.o

    def high(self) -> pd.Series:
        """Return high series.

        Args:
            None
        Returns:
            High series.
        """
        return self.h

    def low(self) -> pd.Series:
        """Return low series.

        Args:
            None
        Returns:
            Low series.
        """
        return self.l

    def close(self) -> pd.Series:
        """Return close series.

        Args:
            None
        Returns:
            Close series.
        """
        return self.c

    def volume(self) -> pd.Series:
        """Return volume series.

        Args:
            None
        Returns:
            Volume series.
        """
        return self.v

    def fng(self) -> pd.DataFrame:
        """Return fng raw, diff1, and 7d z-score as a DataFrame.

        Args:
            None
        Returns:
            DataFrame with fng, fng_diff1, fng_z7d columns.
        """
        fng = self.g
        diff1 = fng.diff()
        roll = fng.rolling(672, min_periods=3)  # 7d = 7*24*4 = 672 bars @15m
        z7d = (fng - roll.mean()) / roll.std().replace(0, np.nan)
        return pd.DataFrame({
            "fng": fng,
            "fng_diff1": diff1.astype("float32"),
            "fng_z7d": z7d.astype("float32"),
        }, index=self.df.index)

    def ttm_trend(self, length: int) -> pd.Series:
        """Compute TTM trend from high/low/close.

        Args:
            length: Window length.
        Returns:
            TTM trend series.
        """
        out = pta.ttm_trend(self.h, self.l, self.c, length=int(length))
        return self._as_series(out)

    def slope(self, length: int) -> pd.Series:
        """Compute slope indicator from close.

        Args:
            length: Window length.
        Returns:
            Slope series.
        """
        out = pta.slope(self.c, length=int(length))
        return self._as_series(out)

    def sma(self, length: int) -> pd.Series:
        """Compute simple moving average.

        Args:
            length: Window length.
        Returns:
            SMA series.
        """
        out = pta.sma(self.c, length=int(length))
        return self._as_series(out)

    def ema(self, length: int) -> pd.Series:
        """Compute exponential moving average.

        Args:
            length: Window length.
        Returns:
            EMA series.
        """
        out = pta.ema(self.c, length=int(length))
        return self._as_series(out)

    def tema(self, length: int) -> pd.Series:
        """Compute triple exponential moving average.

        Args:
            length: Window length.
        Returns:
            TEMA series.
        """
        out = pta.tema(self.c, length=int(length))
        return self._as_series(out)

    def dpo(self, length: int) -> pd.Series:
        """Compute detrended price oscillator.

        Args:
            length: Window length.
        Returns:
            DPO series.
        """
        out = pta.dpo(self.c, length=int(length), centered=False)
        return self._as_series(out)

    def rsi(self, length: int) -> pd.Series:
        """Compute RSI from close.

        Args:
            length: Window length.
        Returns:
            RSI series.
        """
        out = pta.rsi(self.c, length=int(length))
        return self._as_series(out)

    def mom(self, length: int) -> pd.Series:
        """Compute momentum from close.

        Args:
            length: Window length.
        Returns:
            Momentum series.
        """
        out = pta.mom(self.c, length=int(length))
        return self._as_series(out)

    def roc(self, length: int) -> pd.Series:
        """Compute rate of change from close.

        Args:
            length: Window length.
        Returns:
            ROC series.
        """
        out = pta.roc(self.c, length=int(length))
        return self._as_series(out)

    def cti(self, length: int) -> pd.Series:
        """Compute correlation trend indicator from close.

        Args:
            length: Window length.
        Returns:
            CTI series.
        """
        out = pta.cti(self.c, length=int(length))
        return self._as_series(out)

    def cfo(self, length: int) -> pd.Series:
        """Compute Chande forecast oscillator from close.

        Args:
            length: Window length.
        Returns:
            CFO series.
        """
        out = pta.cfo(self.c, length=int(length))
        return self._as_series(out)

    def skew(self, length: int) -> pd.Series:
        """Compute rolling skew from close.

        Args:
            length: Window length.
        Returns:
            Skew series.
        """
        out = pta.skew(self.c, length=int(length))
        return self._as_series(out)

    def kurtosis(self, length: int) -> pd.Series:
        """Compute rolling kurtosis from close.

        Args:
            length: Window length.
        Returns:
            Kurtosis series.
        """
        out = pta.kurtosis(self.c, length=int(length))
        return self._as_series(out)

    def entropy(self, length: int) -> pd.Series:
        """Compute entropy from close.

        Args:
            length: Window length.
        Returns:
            Entropy series.
        """
        out = pta.entropy(self.c, length=int(length))
        return self._as_series(out)

    def decay(self, length: int) -> pd.Series:
        """Compute decay transform from close.

        Args:
            length: Window length.
        Returns:
            Decay series.
        """
        out = pta.decay(self.c, length=int(length))
        return self._as_series(out)

    def decreasing(self, length: int) -> pd.Series:
        """Compute decreasing flag from close.

        Args:
            length: Window length.
        Returns:
            Decreasing series.
        """
        out = pta.decreasing(self.c, length=int(length))
        return self._as_series(out)

    def cmo(self, length: int) -> pd.Series:
        """Compute CMO from close.

        Args:
            length: Window length.
        Returns:
            CMO series.
        """
        out = pta.cmo(self.c, length=int(length))
        return self._as_series(out)

    def bias(self, length: int) -> pd.Series:
        """Compute BIAS from close.

        Args:
            length: Window length.
        Returns:
            BIAS series.
        """
        out = pta.bias(self.c, length=int(length))
        return self._as_series(out, include=("bias",))

    def zscore(self, length: int) -> pd.Series:
        """Compute rolling z-score from close.

        Args:
            length: Window length.
        Returns:
            Z-score series.
        """
        out = pta.zscore(self.c, length=int(length))
        return self._as_series(out)

    def willr(self, length: int) -> pd.Series:
        """Compute Williams %R.

        Args:
            length: Window length.
        Returns:
            WILLR series.
        """
        out = pta.willr(self.h, self.l, self.c, length=int(length))
        return self._as_series(out)

    def cci(self, length: int, c: float = 0.015) -> pd.Series:
        """Compute CCI.

        Args:
            length: Window length.
            c: Scaling constant.
        Returns:
            CCI series.
        """
        out = pta.cci(self.h, self.l, self.c, length=int(length), c=float(c))
        return self._as_series(out)

    def uo(self, fast: int = 7, medium: int = 14, slow: int = 28) -> pd.Series:
        """Compute Ultimate Oscillator.

        Args:
            fast: Fast length.
            medium: Medium length.
            slow: Slow length.
        Returns:
            UO series.
        """
        out = pta.uo(self.h, self.l, self.c, fast=int(fast), medium=int(medium), slow=int(slow))
        return self._as_series(out)

    def atr(self, length: int = 14) -> pd.Series:
        """Compute ATR.

        Args:
            length: ATR length.
        Returns:
            ATR series.
        """
        out = pta.atr(self.h, self.l, self.c, length=int(length))
        return self._as_series(out)

    def massi(self, fast: int = 9, slow: int = 25) -> pd.Series:
        """Compute MASSI.

        Args:
            fast: Fast length.
            slow: Slow length.
        Returns:
            MASSI series.
        """
        out = pta.massi(self.h, self.l, fast=int(fast), slow=int(slow))
        return self._as_series(out)

    def bbp(self, length: int = 20, std: float = 2.0) -> pd.Series:
        """Compute Bollinger %B.

        Args:
            length: Window length.
            std: Standard deviation multiplier.
        Returns:
            BBP series.
        """
        out = pta.bbands(self.c, length=int(length), std=float(std))
        return self._as_series(out, include=("bbp",))

    def vhf(self, length: int) -> pd.Series:
        """Compute VHF.

        Args:
            length: Window length.
        Returns:
            VHF series.
        """
        out = pta.vhf(self.c, length=int(length))
        return self._as_series(out)

    def rwi(self, length: int, line: str = "rwil") -> pd.Series:
        """Compute RWI line from high/low/close.

        Args:
            length: Window length.
            line: Target line, rwih or rwil.
        Returns:
            Selected RWI series.
        """
        out = pta.rwi(self.h, self.l, self.c, length=int(length))
        target = str(line).lower()
        return self._as_series(out, include=(target,))

    def pvr(self) -> pd.Series:
        """Compute PVR.

        Args:
            None
        Returns:
            PVR series.
        """
        out = pta.pvr(self.c, self.v)
        return self._as_series(out)

    def efi(self, length: int = 13) -> pd.Series:
        """Compute EFI.

        Args:
            length: Window length.
        Returns:
            EFI series.
        """
        out = pta.efi(self.c, self.v, length=int(length))
        return self._as_series(out)

    def eom(self, length: int = 14, divisor: float = 100000000.0) -> pd.Series:
        """Compute EOM.

        Args:
            length: Window length.
            divisor: Volume divisor.
        Returns:
            EOM series.
        """
        out = pta.eom(high=self.h, low=self.l, close=self.c, volume=self.v, length=int(length), divisor=float(divisor))
        return self._as_series(out)

    def pvt(self) -> pd.Series:
        """Compute PVT.

        Args:
            None
        Returns:
            PVT series.
        """
        out = pta.pvt(self.c, self.v)
        return self._as_series(out)

    def bop(self) -> pd.Series:
        """Compute BOP.

        Args:
            None
        Returns:
            BOP series.
        """
        out = pta.bop(self.o, self.h, self.l, self.c)
        return self._as_series(out)

    def log_return(self, length: int = 1) -> pd.Series:
        """Compute log return.

        Args:
            length: Return period.
        Returns:
            Log return series.
        """
        out = pta.log_return(self.c, length=int(length))
        return self._as_series(out)

    def percent_return(self, length: int = 1) -> pd.Series:
        """Compute percent return.

        Args:
            length: Return period.
        Returns:
            Percent return series.
        """
        out = pta.percent_return(self.c, length=int(length))
        return self._as_series(out)

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Compute MACD family with stable output naming.

        Args:
            fast: Fast EMA length.
            slow: Slow EMA length.
            signal: Signal EMA length.
        Returns:
            Dataframe with macd/macds/macdh columns.
        """
        out = pta.macd(self.c, fast=int(fast), slow=int(slow), signal=int(signal))
        if out is None or not isinstance(out, pd.DataFrame) or out.empty:
            cols = {
                f"macd_{fast}_{slow}_{signal}": self._nan_series(self.df.index),
                f"macds_{fast}_{slow}_{signal}": self._nan_series(self.df.index),
                f"macdh_{fast}_{slow}_{signal}": self._nan_series(self.df.index),
            }
            return pd.DataFrame(cols, index=self.df.index)

        cols_l = [str(c).lower() for c in out.columns]
        m = self._as_series(out, include=("macd",), exclude=("macds", "macdh"))
        s = self._as_series(out, include=("macds", "signal"))
        h = self._as_series(out, include=("macdh", "hist"))
        if len(cols_l) == 1:
            s = self._nan_series(self.df.index)
            h = self._nan_series(self.df.index)
        return pd.DataFrame(
            {
                f"macd_{fast}_{slow}_{signal}": m,
                f"macds_{fast}_{slow}_{signal}": s,
                f"macdh_{fast}_{slow}_{signal}": h,
            },
            index=self.df.index,
        )

    def stoch(self, k: int = 14, d: int = 3, smooth_k: int = 3) -> pd.DataFrame:
        """Compute stochastic family with stable output naming.

        Args:
            k: K length.
            d: D length.
            smooth_k: Smoothing length for K.
        Returns:
            Dataframe with stochk/stochd columns.
        """
        out = pta.stoch(self.h, self.l, self.c, k=int(k), d=int(d), smooth_k=int(smooth_k))
        k_col = self._as_series(out, include=("stochk",))
        d_col = self._as_series(out, include=("stochd",))
        return pd.DataFrame(
            {
                f"stochk_{k}_{d}_{smooth_k}": k_col,
                f"stochd_{k}_{d}_{smooth_k}": d_col,
            },
            index=self.df.index,
        )

    def adx(self, length: int = 14) -> pd.DataFrame:
        """Compute ADX family with stable output naming.

        Args:
            length: ADX length.
        Returns:
            Dataframe with adx/dmp/dmn columns.
        """
        out = pta.adx(self.h, self.l, self.c, length=int(length))
        adx_col = self._as_series(out, include=("adx",), exclude=("dmp", "dmn"))
        dmp_col = self._as_series(out, include=("dmp",))
        dmn_col = self._as_series(out, include=("dmn",))
        return pd.DataFrame(
            {
                f"adx_{length}": adx_col,
                f"dmp_{length}": dmp_col,
                f"dmn_{length}": dmn_col,
            },
            index=self.df.index,
        )

    def pvo(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Compute PVO family with stable output naming.

        Args:
            fast: Fast EMA length.
            slow: Slow EMA length.
            signal: Signal EMA length.
        Returns:
            Dataframe with pvo/pvos/pvoh columns.
        """
        out = pta.pvo(self.v, fast=int(fast), slow=int(slow), signal=int(signal))
        pvo_col = self._as_series(out, include=("pvo",), exclude=("pvos", "pvoh"))
        pvos_col = self._as_series(out, include=("pvos", "signal"))
        pvoh_col = self._as_series(out, include=("pvoh", "hist"))
        return pd.DataFrame(
            {
                f"pvo_{fast}_{slow}_{signal}": pvo_col,
                f"pvos_{fast}_{slow}_{signal}": pvos_col,
                f"pvoh_{fast}_{slow}_{signal}": pvoh_col,
            },
            index=self.df.index,
        )

    def kvo(self, fast: int = 34, slow: int = 55, signal: int = 13) -> pd.DataFrame:
        """Compute KVO family with stable output naming.

        Args:
            fast: Fast EMA length.
            slow: Slow EMA length.
            signal: Signal EMA length.
        Returns:
            Dataframe with kvo/kvos/kvoh columns.
        """
        out = pta.kvo(self.h, self.l, self.c, self.v, fast=int(fast), slow=int(slow), signal=int(signal))
        kvo_col = self._as_series(out, include=("kvo",), exclude=("kvos", "kvoh"))
        kvos_col = self._as_series(out, include=("kvos", "signal"))
        kvoh_col = self._as_series(out, include=("kvoh", "hist"))
        return pd.DataFrame(
            {
                f"kvo_{fast}_{slow}_{signal}": kvo_col,
                f"kvos_{fast}_{slow}_{signal}": kvos_col,
                f"kvoh_{fast}_{slow}_{signal}": kvoh_col,
            },
            index=self.df.index,
        )

    def aobv(self, mamode: int = 1, length: int = 14) -> pd.DataFrame:
        """Compute AOBV family with stable output naming.

        Args:
            mamode: MA mode.
            length: Base length.
        Returns:
            Dataframe with aobv/aobv_lr/aobv_sr columns.
        """
        out = pta.aobv(self.c, self.v, mamode=int(mamode), length=int(length))
        if out is None or not isinstance(out, pd.DataFrame) or out.empty:
            cols = {
                f"aobv_{mamode}_{length}": self._nan_series(self.df.index),
                f"aobv_lr_{mamode}_{length}": self._nan_series(self.df.index),
                f"aobv_sr_{mamode}_{length}": self._nan_series(self.df.index),
            }
            if int(mamode) == 1 and int(length) == 14:
                cols["aobv"] = cols[f"aobv_{mamode}_{length}"]
                cols["aobv_lr"] = cols[f"aobv_lr_{mamode}_{length}"]
                cols["aobv_sr"] = cols[f"aobv_sr_{mamode}_{length}"]
            return pd.DataFrame(cols, index=self.df.index)

        base = self._as_series(out, include=("aobv",), exclude=("lr", "sr"))
        lr = self._as_series(out, include=("lr",))
        sr = self._as_series(out, include=("sr",))
        cols = {
            f"aobv_{mamode}_{length}": base,
            f"aobv_lr_{mamode}_{length}": lr,
            f"aobv_sr_{mamode}_{length}": sr,
        }
        if int(mamode) == 1 and int(length) == 14:
            cols["aobv"] = cols[f"aobv_{mamode}_{length}"]
            cols["aobv_lr"] = cols[f"aobv_lr_{mamode}_{length}"]
            cols["aobv_sr"] = cols[f"aobv_sr_{mamode}_{length}"]
        return pd.DataFrame(cols, index=self.df.index)

    def rvi(self, length: int) -> pd.Series:
        """Compute Relative Vigor Index from close.

        Args:
            length: Window length.
        Returns:
            RVI series.
        """
        out = pta.rvi(self.c, length=int(length))
        if isinstance(out, pd.DataFrame):
            return self._as_series(out, include=("rvi",))
        return self._as_series(out)

    def kdj(self, k: int = 9, d: int = 3, smooth_k: int = 3) -> pd.Series:
        """Compute KDJ J-line (3K - 2D) derived from stochastic.

        Args:
            k: K length.
            d: D length.
            smooth_k: Smoothing length for K.
        Returns:
            J-line series.
        """
        st = pta.stoch(self.h, self.l, self.c, k=int(k), d=int(d), smooth_k=int(smooth_k))
        k_col = self._as_series(st, include=("stochk",))
        d_col = self._as_series(st, include=("stochd",))
        return 3 * k_col - 2 * d_col

    def truerange(self) -> pd.Series:
        """Compute single-period True Range.

        Args:
            None
        Returns:
            True Range series.
        """
        out = pta.true_range(self.h, self.l, self.c)
        return self._as_series(out)

    def atrp(self, length: int = 14) -> pd.Series:
        """Compute ATR as percentage of close (ATR / |close|).

        Args:
            length: ATR length.
        Returns:
            ATRP series.
        """
        atr_val = pta.atr(self.h, self.l, self.c, length=int(length))
        atr_s = self._as_series(atr_val)
        c_abs = self.c.abs().replace(0, np.nan)
        return atr_s / c_abs

    def hl_range(self, window: int, pct: bool = True) -> pd.Series:
        """Compute rolling mean of high-low range.

        Args:
            window: Rolling window size.
            pct: If True, normalize by |close|.
        Returns:
            HL range mean series.
        """
        hl = self.h - self.l
        w = int(window)
        if pct:
            base = self.c.abs().replace(0, np.nan)
            s = (hl / base).rolling(w, min_periods=max(1, w // 2)).mean()
        else:
            s = hl.rolling(w, min_periods=max(1, w // 2)).mean()
        return s

    def ewmret(self, halflife: int) -> pd.DataFrame:
        """Compute EWM mean and std of log returns.

        Args:
            halflife: EWM halflife in bars.
        Returns:
            DataFrame with ewm_m_{halflife} and ewm_s_{halflife} columns.
        """
        hl = int(halflife)
        lr = np.log(self.c).diff()
        m = lr.ewm(halflife=hl, adjust=False).mean()
        s = lr.ewm(halflife=hl, adjust=False).std()
        return pd.DataFrame({
            f"ewm_m_{hl}": m,
            f"ewm_s_{hl}": s,
        }, index=self.df.index)

    def pxvol(self) -> pd.DataFrame:
        """Compute price-volume interaction features.

        Args:
            None
        Returns:
            DataFrame with dir_strength, pxv_lr_vchg, dirxvol columns.
        """
        eps = 1e-9
        dir_strength = (self.c - self.o) / np.maximum(self.h - self.l, eps)
        logret = np.log(self.c).diff()
        vol_chg = self.v.pct_change().fillna(0.0).clip(-1.0, 1.0)
        return pd.DataFrame({
            "dir_strength": dir_strength,
            "pxv_lr_vchg": logret * vol_chg,
            "dirxvol": dir_strength * self.v,
        }, index=self.df.index)

    def time_cyc(self, tz: str = "UTC", daily: bool = True, weekly: bool = True) -> pd.DataFrame:
        """Compute sine/cosine time-of-day and day-of-week features.

        Args:
            tz: Target timezone for hour/dow extraction.
            daily: Whether to include time-of-day features.
            weekly: Whether to include day-of-week features.
        Returns:
            DataFrame with tod_sin, tod_cos, dow_sin, dow_cos columns.
        """
        idx = pd.DatetimeIndex(self.df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx = idx.tz_convert(tz)
        out = {}
        if daily:
            hour = idx.hour + idx.minute / 60.0
            out["tod_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
            out["tod_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
        if weekly:
            dow = idx.dayofweek.astype(float)
            out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
            out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
        return pd.DataFrame(out, index=self.df.index)

    def amat(self, fast: int = 8, slow: int = 21, mamode: int = 2) -> pd.DataFrame:
        """Compute AMAT trend signal LR component.

        Args:
            fast: Fast MA length.
            slow: Slow MA length.
            mamode: MA mode (pandas_ta convention).
        Returns:
            DataFrame with amat_lr_{fast}_{slow}_{mamode} column.
        """
        out = pta.amat(self.c, fast=int(fast), slow=int(slow), mamode=int(mamode))
        if out is None or not isinstance(out, pd.DataFrame) or out.empty:
            return pd.DataFrame({
                f"amat_lr_{fast}_{slow}_{mamode}": self._nan_series(self.df.index),
            }, index=self.df.index)
        lr_col = self._as_series(out, include=("lr",))
        return pd.DataFrame({
            f"amat_lr_{fast}_{slow}_{mamode}": lr_col,
        }, index=self.df.index)

    @staticmethod
    def load_feature_list(path: Path | str) -> list[str]:
        """Load feature names from a txt file.

        Args:
            path: Txt path containing one feature name per line.
        Returns:
            Ordered feature name list.
        """
        txt = Path(path)
        if not txt.exists():
            raise FileNotFoundError(f"feature list not found: {txt}")
        out: list[str] = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
        return out

    @staticmethod
    def _parse_ints(tail: str) -> tuple[int, ...] | None:
        """Parse underscore-separated integers.

        Args:
            tail: Tail string after base feature name.
        Returns:
            Integer tuple or None.
        """
        if not tail:
            return tuple()
        parts = tail.split("_")
        vals: list[int] = []
        for p in parts:
            if not re.fullmatch(r"\d+", p):
                return None
            vals.append(int(p))
        return tuple(vals)

    def _parse_feature_name(self, name: str) -> dict[str, Any] | None:
        """Parse canonical feature name into dispatch spec.

        Args:
            name: Feature name from txt list.
        Returns:
            Dispatch spec dictionary or None for unknown names.
        """
        key = str(name).strip().lower()
        if key in self._RAW_COLUMNS:
            return {"kind": "raw", "name": key}

        m = re.fullmatch(r"(macd|macds|macdh)_(\d+)_(\d+)_(\d+)", key)
        if m:
            return {
                "kind": "multi",
                "family": "macd",
                "output": key,
                "kwargs": {"fast": int(m.group(2)), "slow": int(m.group(3)), "signal": int(m.group(4))},
            }

        m = re.fullmatch(r"(stochk|stochd)_(\d+)_(\d+)_(\d+)", key)
        if m:
            return {
                "kind": "multi",
                "family": "stoch",
                "output": key,
                "kwargs": {"k": int(m.group(2)), "d": int(m.group(3)), "smooth_k": int(m.group(4))},
            }

        m = re.fullmatch(r"(adx|dmp|dmn)_(\d+)", key)
        if m:
            return {
                "kind": "multi",
                "family": "adx",
                "output": key,
                "kwargs": {"length": int(m.group(2))},
            }

        m = re.fullmatch(r"(pvo|pvos|pvoh)_(\d+)_(\d+)_(\d+)", key)
        if m:
            return {
                "kind": "multi",
                "family": "pvo",
                "output": key,
                "kwargs": {"fast": int(m.group(2)), "slow": int(m.group(3)), "signal": int(m.group(4))},
            }

        m = re.fullmatch(r"(kvo|kvos|kvoh)_(\d+)_(\d+)_(\d+)", key)
        if m:
            return {
                "kind": "multi",
                "family": "kvo",
                "output": key,
                "kwargs": {"fast": int(m.group(2)), "slow": int(m.group(3)), "signal": int(m.group(4))},
            }

        m = re.fullmatch(r"(aobv|aobv_lr|aobv_sr)_(\d+)_(\d+)", key)
        if m:
            return {
                "kind": "multi",
                "family": "aobv",
                "output": key,
                "kwargs": {"mamode": int(m.group(2)), "length": int(m.group(3))},
            }

        for raw in ("aobv", "aobv_lr", "aobv_sr"):
            if key == raw:
                return {
                    "kind": "multi",
                    "family": "aobv",
                    "output": raw,
                    "kwargs": {"mamode": 1, "length": 14},
                }

        m = re.fullmatch(r"bbp_(\d+)_([0-9]+(?:\.[0-9]+)?)", key)
        if m:
            return {
                "kind": "single",
                "method": "bbp",
                "output": key,
                "kwargs": {"length": int(m.group(1)), "std": float(m.group(2))},
            }

        m = re.fullmatch(r"(uo)_(\d+)_(\d+)_(\d+)", key)
        if m:
            return {
                "kind": "single",
                "method": "uo",
                "output": key,
                "kwargs": {"fast": int(m.group(2)), "medium": int(m.group(3)), "slow": int(m.group(4))},
            }

        m = re.fullmatch(r"massi_(\d+)_(\d+)", key)
        if m:
            return {
                "kind": "single",
                "method": "massi",
                "output": key,
                "kwargs": {"fast": int(m.group(1)), "slow": int(m.group(2))},
            }

        m = re.fullmatch(r"(rwih|rwil|rwi)_(\d+)", key)
        if m:
            line = m.group(1)
            if line == "rwi":
                line = "rwil"
            return {
                "kind": "single",
                "method": "rwi",
                "output": key,
                "kwargs": {"length": int(m.group(2)), "line": line},
            }

        m = re.fullmatch(r"(logret|pctret)_(\d+)", key)
        if m:
            method = "log_return" if m.group(1) == "logret" else "percent_return"
            return {
                "kind": "single",
                "method": method,
                "output": key,
                "kwargs": {"length": int(m.group(2))},
            }

        scalar_roots = (
            "ttm_trend",
            "slope",
            "sma",
            "ema",
            "tema",
            "dpo",
            "rsi",
            "mom",
            "roc",
            "cti",
            "cfo",
            "skew",
            "kurtosis",
            "entropy",
            "decay",
            "decreasing",
            "cmo",
            "bias",
            "zscore",
            "willr",
            "cci",
            "atr",
            "vhf",
            "rvi",
            "atrp",
            "efi",
            "eom",
        )
        for root in sorted(scalar_roots, key=len, reverse=True):
            if key.startswith(f"{root}_"):
                tail = key[len(root) + 1 :]
                ints = self._parse_ints(tail)
                if ints is None:
                    return None
                if root in {"cci"}:
                    if len(ints) != 1:
                        return None
                    return {
                        "kind": "single",
                        "method": root,
                        "output": key,
                        "kwargs": {"length": ints[0]},
                    }
                if len(ints) != 1:
                    return None
                return {
                    "kind": "single",
                    "method": root,
                    "output": key,
                    "kwargs": {"length": ints[0]},
                }

        if key in {"pvr", "pvt", "bop"}:
            return {"kind": "single", "method": key, "output": key, "kwargs": {}}
        if key == "efi":
            return {"kind": "single", "method": key, "output": key, "kwargs": {"length": 13}}
        if key == "eom":
            return {
                "kind": "single",
                "method": key,
                "output": key,
                "kwargs": {"length": 14, "divisor": 100000000.0},
            }
        if key == "truerange":
            return {"kind": "single", "method": "truerange", "output": key, "kwargs": {}}

        if key in {"fng", "fng_diff1", "fng_z7d"}:
            return {
                "kind": "multi",
                "family": "fng",
                "output": key,
                "kwargs": {},
            }

        # KDJ: kdj_{k}_{d}_{smooth_k}
        m = re.fullmatch(r"kdj_(\d+)_(\d+)_(\d+)", key)
        if m:
            return {
                "kind": "single",
                "method": "kdj",
                "output": key,
                "kwargs": {"k": int(m.group(1)), "d": int(m.group(2)), "smooth_k": int(m.group(3))},
            }

        # HL_RANGE: hl_range_{window}
        m = re.fullmatch(r"hl_range_(\d+)", key)
        if m:
            return {
                "kind": "single",
                "method": "hl_range",
                "output": key,
                "kwargs": {"window": int(m.group(1))},
            }

        # AMAT: amat_lr_{fast}_{slow}_{mamode}
        m = re.fullmatch(r"amat_lr_(\d+)_(\d+)_(\d+)", key)
        if m:
            return {
                "kind": "multi",
                "family": "amat",
                "output": key,
                "kwargs": {"fast": int(m.group(1)), "slow": int(m.group(2)), "mamode": int(m.group(3))},
            }

        # EWMRET: ewm_m_{hl} / ewm_s_{hl}
        m = re.fullmatch(r"(ewm_m|ewm_s)_(\d+)", key)
        if m:
            return {
                "kind": "multi",
                "family": "ewmret",
                "output": key,
                "kwargs": {"halflife": int(m.group(2))},
            }

        # PXVOL: dir_strength / pxv_lr_vchg / dirxvol
        if key in {"dir_strength", "pxv_lr_vchg", "dirxvol"}:
            return {
                "kind": "multi",
                "family": "pxvol",
                "output": key,
                "kwargs": {},
            }

        # TIME_CYC: tod_sin / tod_cos / dow_sin / dow_cos
        if key in {"tod_sin", "tod_cos", "dow_sin", "dow_cos"}:
            return {
                "kind": "multi",
                "family": "time_cyc",
                "output": key,
                "kwargs": {},
            }

        return None

    @staticmethod
    def _cache_key(family: str, kwargs: dict[str, Any]) -> tuple[str, tuple[tuple[str, Any], ...]]:
        """Build deterministic cache key for family computation.

        Args:
            family: Family name.
            kwargs: Keyword parameters.
        Returns:
            Hashable cache key.
        """
        items = tuple(sorted(kwargs.items(), key=lambda kv: kv[0]))
        return family, items

    def compute_from_list(
        self,
        feature_names: Sequence[str],
        *,
        strict: bool = True,
        fillna_value: float = 0.0,
        dtype: str = "float32",
    ) -> tuple[pd.DataFrame, list[str]]:
        """Compute selected features by canonical feature-name list.

        Args:
            feature_names: Ordered feature names.
            strict: Whether unknown feature should raise error.
            fillna_value: Fill value for NaN.
            dtype: Output dtype.
        Returns:
            Tuple of computed feature dataframe and skipped unknown names.
        """
        unique_names: list[str] = []
        seen: set[str] = set()
        for raw in feature_names:
            name = str(raw).strip().lower()
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            unique_names.append(name)

        out: dict[str, pd.Series] = {}
        skipped: list[str] = []
        cache: dict[tuple[str, tuple[tuple[str, Any], ...]], pd.DataFrame] = {}

        for name in unique_names:
            spec = self._parse_feature_name(name)
            if spec is None:
                if strict:
                    raise ValueError(f"Unknown feature name in list: {name}")
                skipped.append(name)
                continue

            kind = spec["kind"]
            if kind == "raw":
                col = spec["name"]
                s = pd.Series(self.df[col], index=self.df.index)
                out[name] = self._sanitize_feature(s, fillna_value=fillna_value, dtype=dtype)
                continue

            if kind == "single":
                method = str(spec["method"])
                kwargs = dict(spec.get("kwargs", {}))
                output_name = str(spec["output"])
                fn = getattr(self, method, None)
                if fn is None:
                    if strict:
                        raise ValueError(f"Feature method not found: {method} (from {name})")
                    skipped.append(name)
                    continue
                s = fn(**kwargs)
                out[output_name] = self._sanitize_feature(s, fillna_value=fillna_value, dtype=dtype)
                continue

            family = str(spec["family"])
            kwargs = dict(spec.get("kwargs", {}))
            output_name = str(spec["output"])
            key = self._cache_key(family, kwargs)
            if key not in cache:
                fn = getattr(self, family, None)
                if fn is None:
                    if strict:
                        raise ValueError(f"Feature family method not found: {family} (from {name})")
                    skipped.append(name)
                    continue
                df_multi = fn(**kwargs)
                if not isinstance(df_multi, pd.DataFrame):
                    if strict:
                        raise ValueError(f"Feature family output must be DataFrame: {family}")
                    skipped.append(name)
                    continue
                cache[key] = self._sanitize_frame(df_multi, fillna_value=fillna_value, dtype=dtype)

            df_cached = cache[key]
            if output_name not in df_cached.columns:
                if strict:
                    raise ValueError(
                        f"Feature output not found in family result: {output_name} (family={family}, kwargs={kwargs})"
                    )
                skipped.append(name)
                continue
            out[output_name] = pd.Series(df_cached[output_name], index=self.df.index)

        df_out = pd.DataFrame(out, index=self.df.index)
        dups = df_out.columns[df_out.columns.duplicated()].tolist()
        if dups:
            raise ValueError(f"Duplicate output columns after compute_from_list: {dups}")
        return df_out, skipped

    def compute_from_txt(
        self,
        txt_path: Path | str,
        *,
        strict: bool = True,
        fillna_value: float = 0.0,
        dtype: str = "float32",
    ) -> tuple[pd.DataFrame, list[str]]:
        """Compute selected features from txt file.

        Args:
            txt_path: Path to feature list txt.
            strict: Whether unknown feature should raise error.
            fillna_value: Fill value for NaN.
            dtype: Output dtype.
        Returns:
            Tuple of computed feature dataframe and skipped unknown names.
        """
        names = self.load_feature_list(txt_path)
        return self.compute_from_list(
            names,
            strict=strict,
            fillna_value=fillna_value,
            dtype=dtype,
        )

    @staticmethod
    def _unique_positive_ints(values: Sequence[int]) -> list[int]:
        """Normalize lengths into sorted positive unique integers.

        Args:
            values: Raw lengths.
        Returns:
            Normalized length list.
        """
        out: set[int] = set()
        for v in values:
            iv = int(v)
            if iv > 0:
                out.add(iv)
        return sorted(out)

    def compute_all(
        self,
        *,
        len_fast: Sequence[int] = (4, 16),
        len_trend: Sequence[int] = (16, 48, 96),
        len_stats: Sequence[int] = (48, 96, 192),
        logret_lengths: Sequence[int] = (1, 4, 8, 16),
        pctret_lengths: Sequence[int] = (1, 4, 8, 16),
        include_raw: bool = True,
        fillna_value: float = 0.0,
        dtype: str = "float32",
    ) -> pd.DataFrame:
        """Compute all supported PTA features with canonical names.

        Uses the same grouping as ``_build_features`` in
        ``feat_computer_pta.py`` (len_fast / len_trend / len_stats).

        Args:
            len_fast: Lengths for momentum / oscillator group.
            len_trend: Lengths for trend / strength group.
            len_stats: Lengths for statistics group.
            logret_lengths: Lengths for log returns.
            pctret_lengths: Lengths for percent returns.
            include_raw: Whether to include open/high/low/close/volume/fng.
            fillna_value: Fill value for NaN.
            dtype: Output dtype.
        Returns:
            Dataframe with all configured canonical features.
        """
        feat_names: list[str] = []

        # ── Raw ──
        if include_raw:
            feat_names.extend(["open", "high", "low", "close", "volume"])
            feat_names.extend(["fng", "fng_diff1", "fng_z7d"])

        # ── Momentum / Oscillator (len_fast) ──
        momentum_roots = ("rsi", "willr", "cmo", "cfo", "roc", "mom", "rvi")
        fast = self._unique_positive_ints(len_fast)
        for l in fast:
            for root in momentum_roots:
                feat_names.append(f"{root}_{l}")

        # KDJ
        for k, d, sk in [(9, 3, 3), (16, 3, 3)]:
            feat_names.append(f"kdj_{k}_{d}_{sk}")

        # Stoch (same lengths as fast)
        for k in fast:
            feat_names.extend([f"stochk_{k}_3_3", f"stochd_{k}_3_3"])

        # ── Trend / Strength (len_trend) ──
        trend_roots = ("ttm_trend", "bias", "slope", "vhf", "sma", "ema", "tema", "dpo")
        trend = self._unique_positive_ints(len_trend)
        for l in trend:
            for root in trend_roots:
                feat_names.append(f"{root}_{l}")
            feat_names.extend([f"adx_{l}", f"dmp_{l}", f"dmn_{l}"])

        # MACD
        for f, s, sig in [(12, 26, 9), (4, 12, 6)]:
            feat_names.extend([f"macd_{f}_{s}_{sig}", f"macds_{f}_{s}_{sig}", f"macdh_{f}_{s}_{sig}"])

        # AMAT LR
        for f, s, m in [(8, 21, 2)]:
            feat_names.append(f"amat_lr_{f}_{s}_{m}")

        # ── Statistics (len_stats) ──
        stats_roots = ("entropy", "skew", "kurtosis")
        stats = self._unique_positive_ints(len_stats)
        for l in stats:
            for root in stats_roots:
                feat_names.append(f"{root}_{l}")

        # Pattern / filter
        feat_names.extend(["decreasing_4", "decreasing_16", "decay_16", "decay_96"])

        # ── Volatility ──
        feat_names.append("truerange")
        for l in [14, 28]:
            feat_names.extend([f"atr_{l}", f"atrp_{l}"])
        for w in [16, 32]:
            feat_names.append(f"hl_range_{w}")
        feat_names.append("massi_9_25")
        feat_names.append("bbp_16_2.0")

        # EWMRET
        for hl in [4, 12, 48]:
            feat_names.extend([f"ewm_m_{hl}", f"ewm_s_{hl}"])

        # ── Volume ──
        for f, s, sig in [(12, 26, 9), (24, 52, 18)]:
            feat_names.extend([f"pvo_{f}_{s}_{sig}", f"pvos_{f}_{s}_{sig}", f"pvoh_{f}_{s}_{sig}"])
        feat_names.extend(["pvr", "bop"])
        for f, s, sig in [(34, 55, 13), (21, 34, 13)]:
            feat_names.extend([f"kvo_{f}_{s}_{sig}", f"kvos_{f}_{s}_{sig}", f"kvoh_{f}_{s}_{sig}"])
        for l in [14, 32, 96]:
            feat_names.extend([f"efi_{l}", f"eom_{l}"])

        # PXVOL
        feat_names.extend(["dir_strength", "pxv_lr_vchg", "dirxvol"])

        # ── Returns ──
        for l in self._unique_positive_ints(logret_lengths):
            feat_names.append(f"logret_{l}")
        for l in self._unique_positive_ints(pctret_lengths):
            feat_names.append(f"pctret_{l}")

        # ── Time cycle ──
        feat_names.extend(["tod_sin", "tod_cos", "dow_sin", "dow_cos"])

        df_all, _ = self.compute_from_list(
            feat_names,
            strict=True,
            fillna_value=fillna_value,
            dtype=dtype,
        )
        return df_all


"""
API usage

1) 直接一條條呼叫單指標:
    lib = FeatureLibPTA(df_raw)
    rsi_s = lib.rsi(14)
    macd_df = lib.macd(12, 26, 9)
    # caller 統一做 shift
    rsi_s = rsi_s.shift(1).fillna(0.0).astype("float32")
    macd_df = macd_df.shift(1).fillna(0.0).astype("float32")

2) 計算全部支援特徵:
    lib = FeatureLibPTA(df_raw)
    feat_df = lib.compute_all()
    feat_df = feat_df.shift(1).fillna(0.0).astype("float32")

3) 從 txt 計算:
    lib = FeatureLibPTA(df_raw)
    feat_df, skipped = lib.compute_from_txt(
        "path/to/feature_list.txt",
        strict=False,  # train 建議 True, runtime 可 False
    )
    feat_df = feat_df.shift(1).fillna(0.0).astype("float32")

4) 從 list 計算:
    lib = FeatureLibPTA(df_raw)
    feat_df, skipped = lib.compute_from_list(
        ["open", "close", "rsi_14", "macd_12_26_9"],
        strict=True,
    )
    feat_df = feat_df.shift(1).fillna(0.0).astype("float32")
"""
