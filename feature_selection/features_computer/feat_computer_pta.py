from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from feature_selection.features_computer.feat_lib_pta import FeatureLibPTA


class FeatureComputerPTA:
    """Feature computer based on FeatureLibPTA and feature txt plans.

    Args:
        cfg: Runtime configuration.
    Returns:
        None
    """

    REQUIRED_COLUMNS = ("datetime", "timestamp", "open", "high", "low", "close", "volume", "fng")

    def __init__(self, cfg: dict[str, Any] | None = None):
        """Initialize runtime options for PTA feature computation.

        Args:
            cfg: Configuration dictionary.
        Returns:
            None
        """
        self.cfg = dict(cfg or {})
        self.data_cfg = dict(self.cfg.get("data", {}) or {})
        self.default_feat_plan = self.cfg.get("feat_plan", self.cfg.get("feat_txt_path", "all"))
        self.strict_txt = bool(self.cfg.get("strict_txt", True))
        self.shift_bars = int(self.cfg.get("shift_bars", 1))
        self.fill_nan_mode = str(self.cfg.get("fill_nan", "last")).strip().lower()

        norm_cfg = dict(self.cfg.get("normalization", {}) or {})
        self.norm_mode = str(norm_cfg.get("mode", "none")).strip().lower()
        self.norm_window = norm_cfg.get("rolling_window")
        self.norm_min_periods = int(norm_cfg.get("min_periods", 1))
        self.norm_std_floor = float(norm_cfg.get("std_floor", 1e-8))

        self.last_skipped: list[str] = []

    def _load_input_df(self, df_raw: pd.DataFrame | None = None) -> pd.DataFrame:
        """Load source dataframe from argument or configured csv path.

        Args:
            df_raw: Optional in-memory source dataframe.
        Returns:
            Loaded dataframe copy.
        """
        if df_raw is not None:
            if not isinstance(df_raw, pd.DataFrame):
                raise TypeError("df_raw must be a pandas DataFrame.")
            return df_raw.copy()

        path = self.data_cfg.get("ohlcv_fng_path") or self.cfg.get("input_path")
        if not path:
            raise ValueError("Missing input path: cfg.data.ohlcv_fng_path or cfg.input_path is required.")
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input csv not found: {csv_path}")
        return pd.read_csv(csv_path)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dataframe column names to lowercase stripped names.

        Args:
            df: Input dataframe.
        Returns:
            Dataframe with normalized column names.
        """
        out = df.copy()
        out.columns = [str(c).strip().lower() for c in out.columns]
        return out

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Validate required input schema columns.

        Args:
            df: Input dataframe.
        Returns:
            None
        """
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @staticmethod
    def _to_datetime(col: pd.Series) -> pd.DatetimeIndex:
        """Convert a datetime-like series to UTC datetime index.

        Args:
            col: Source datetime-like series.
        Returns:
            UTC datetime index.
        """
        return pd.DatetimeIndex(pd.to_datetime(col, errors="coerce", utc=True))

    @staticmethod
    def _timestamp_to_datetime(col: pd.Series) -> pd.DatetimeIndex:
        """Convert numeric timestamp series to UTC datetime index.

        Args:
            col: Numeric timestamp series in seconds or milliseconds.
        Returns:
            UTC datetime index.
        """
        ts = pd.to_numeric(col, errors="coerce")
        sample = ts.dropna()
        unit = "ms"
        if not sample.empty and sample.iloc[0] < 1_000_000_000_000:
            unit = "s"
        return pd.DatetimeIndex(pd.to_datetime(ts, unit=unit, errors="coerce", utc=True))

    def _to_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build UTC datetime index from datetime/timestamp columns.

        Args:
            df: Input dataframe.
        Returns:
            Indexed dataframe sorted by datetime.
        """
        out = df.copy()
        idx = self._to_datetime(out["datetime"])
        if idx.isna().all():
            idx = self._timestamp_to_datetime(out["timestamp"])
        if idx.isna().any():
            raise ValueError("datetime/timestamp contains unparsable values.")

        out.index = idx
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out

    @staticmethod
    def _coerce_required_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce required numeric columns to float.

        Args:
            df: Input dataframe.
        Returns:
            Coerced dataframe.
        """
        out = df.copy()
        for col in ("open", "high", "low", "close", "volume", "fng"):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        return out

    @staticmethod
    def _dedup_keep_order(cols: Sequence[str]) -> list[str]:
        """Deduplicate column names while preserving the first order.

        Args:
            cols: Input column sequence.
        Returns:
            Deduplicated ordered column list.
        """
        seen: set[str] = set()
        out: list[str] = []
        for c in cols:
            c = str(c)
            if c in seen:
                continue
            seen.add(c)
            out.append(c)
        return out

    @staticmethod
    def _extract_main_col(df_multi: pd.DataFrame, prefix: str) -> pd.Series:
        """Extract the main line column from a multi-output indicator result.

        Args:
            df_multi: Multi-output dataframe.
            prefix: Prefix of the main line.
        Returns:
            Selected main line series.
        """
        cols = [str(c) for c in df_multi.columns]
        if prefix == "pvo":
            names = [c for c in cols if c.startswith("pvo_") and not c.startswith("pvos_") and not c.startswith("pvoh_")]
        elif prefix == "kvo":
            names = [c for c in cols if c.startswith("kvo_") and not c.startswith("kvos_") and not c.startswith("kvoh_")]
        else:
            names = [c for c in cols if c.startswith(f"{prefix}_")]
        target = names[0] if names else cols[0]
        return pd.Series(df_multi[target], index=df_multi.index)

    def _build_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Build default all-style features using loop-based groups.

        Args:
            dataframe: Input normalized dataframe.
        Returns:
            Feature dataframe.
        """
        len_fast = [4, 16]
        len_trend = [16, 48, 96]
        len_stats = [48, 96, 192]

        lib = FeatureLibPTA(dataframe.copy())
        feat_cols: list[str] = []

        # Raw group
        for name, fn in (
            ("open", lib.open),
            ("high", lib.high),
            ("low", lib.low),
            ("close", lib.close),
            ("volume", lib.volume),
        ):
            lib.add_series(name, fn(), fillna_value=None, dtype=None, shift=0)
            feat_cols.append(name)

        # FNG group (raw + diff1 + z7d)
        fng_df = lib._sanitize_frame(lib.fng(), fillna_value=None, dtype=None, shift=0)
        for col in fng_df.columns:
            lib.df[col] = fng_df[col]
            feat_cols.append(col)

        # Momentum / Oscillator group
        for fn in [
            lib.rsi,
            lib.willr,
            lib.cmo,
            lib.cfo,
            lib.roc,
            lib.mom,
            lib.rvi,
        ]:
            lib.add_feat(fn, lengths=len_fast, fillna_value=None, dtype=None, shift=0)
            for l in len_fast:
                feat_cols.append(f"{fn.__name__}_{int(l)}")

        # KDJ group
        for k, d, sk in [(9, 3, 3), (16, 3, 3)]:
            name = f"kdj_{k}_{d}_{sk}"
            lib.add_series(name, lib.kdj(k=k, d=d, smooth_k=sk), fillna_value=None, dtype=None, shift=0)
            feat_cols.append(name)

        # Trend / Strength group
        for l in len_trend:
            adx_df = lib.adx(length=l)
            adx_name = f"adx_{int(l)}"
            lib.add_series(adx_name, self._extract_main_col(adx_df, "adx"), fillna_value=None, dtype=None, shift=0)
            feat_cols.append(adx_name)
        for fn in [lib.ttm_trend, lib.bias, lib.slope, lib.vhf]:
            lib.add_feat(fn, lengths=len_trend, fillna_value=None, dtype=None, shift=0)
            for l in len_trend:
                feat_cols.append(f"{fn.__name__}_{int(l)}")

        # AMAT LR group
        for f, s, m in [(8, 21, 2)]:
            amat_df = lib.amat(fast=f, slow=s, mamode=m)
            amat_san = lib._sanitize_frame(amat_df, fillna_value=None, dtype=None, shift=0)
            for col in amat_san.columns:
                lib.df[col] = amat_san[col]
                feat_cols.append(col)

        # Statistic group
        for fn in [lib.entropy,
                   lib.skew,
                   lib.kurtosis]:
            lib.add_feat(fn, lengths=len_stats, fillna_value=None, dtype=None, shift=0)
            for l in len_stats:
                feat_cols.append(f"{fn.__name__}_{int(l)}")

        # Pattern / filter group
        lib.add_feat(lib.decreasing, lengths=[4, 16], fillna_value=None, dtype=None, shift=0)
        feat_cols.extend(["decreasing_4", "decreasing_16"])
        lib.add_feat(lib.decay, lengths=[16, 96], fillna_value=None, dtype=None, shift=0)
        feat_cols.extend(["decay_16", "decay_96"])

        # Volatility group
        lib.add_series("truerange", lib.truerange(), fillna_value=None, dtype=None, shift=0)
        feat_cols.append("truerange")
        for l in [14, 28]:
            lib.add_series(f"atr_{l}", lib.atr(length=l), fillna_value=None, dtype=None, shift=0)
            lib.add_series(f"atrp_{l}", lib.atrp(length=l), fillna_value=None, dtype=None, shift=0)
            feat_cols.extend([f"atr_{l}", f"atrp_{l}"])
        for w in [16, 32]:
            lib.add_series(f"hl_range_{w}", lib.hl_range(window=w), fillna_value=None, dtype=None, shift=0)
            feat_cols.append(f"hl_range_{w}")
        lib.add_series("massi_9_25", lib.massi(fast=9, slow=25), fillna_value=None, dtype=None, shift=0)
        feat_cols.append("massi_9_25")
        lib.add_series("bbp_16_2.0", lib.bbp(length=16, std=2.0), fillna_value=None, dtype=None, shift=0)
        feat_cols.append("bbp_16_2.0")

        # EWMRET group
        for hl in [4, 12, 48]:
            ewm_df = lib._sanitize_frame(lib.ewmret(halflife=hl), fillna_value=None, dtype=None, shift=0)
            for col in ewm_df.columns:
                lib.df[col] = ewm_df[col]
                feat_cols.append(col)

        # Volume group
        for f, s, sig in [(12, 26, 9), (24, 52, 18)]:
            pvo_df = lib.pvo(fast=f, slow=s, signal=sig)
            pvo_name = f"pvo_{int(f)}_{int(s)}_{int(sig)}"
            lib.add_series(pvo_name, self._extract_main_col(pvo_df, "pvo"), fillna_value=None, dtype=None, shift=0)
            feat_cols.append(pvo_name)
        lib.add_series("pvr", lib.pvr(), fillna_value=None, dtype=None, shift=0)
        feat_cols.append("pvr")
        lib.add_series("bop", lib.bop(), fillna_value=None, dtype=None, shift=0)
        feat_cols.append("bop")

        for f, s, sig in [(34, 55, 13), (21, 34, 13)]:
            kvo_df = lib.kvo(fast=f, slow=s, signal=sig)
            kvo_name = f"kvo_{int(f)}_{int(s)}_{int(sig)}"
            lib.add_series(kvo_name, self._extract_main_col(kvo_df, "kvo"), fillna_value=None, dtype=None, shift=0)
            feat_cols.append(kvo_name)

        for l in [14, 32, 96]:
            lib.add_series(f"efi_{int(l)}", lib.efi(length=l), fillna_value=None, dtype=None, shift=0)
            lib.add_series(f"eom_{int(l)}", lib.eom(length=l), fillna_value=None, dtype=None, shift=0)
            feat_cols.extend([f"efi_{int(l)}", f"eom_{int(l)}"])

        # Price-volume interaction group
        pxvol_df = lib._sanitize_frame(lib.pxvol(), fillna_value=None, dtype=None, shift=0)
        for col in pxvol_df.columns:
            lib.df[col] = pxvol_df[col]
            feat_cols.append(col)

        # Returns group
        for lag in [1, 4, 8, 16]:
            lib.add_series(f"logret_{lag}", lib.log_return(length=lag), fillna_value=None, dtype=None, shift=0)
            feat_cols.append(f"logret_{lag}")

        # Time cycle group
        tcyc_df = lib._sanitize_frame(
            lib.time_cyc(tz="Asia/Taipei", daily=True, weekly=True),
            fillna_value=None, dtype=None, shift=0,
        )
        for col in tcyc_df.columns:
            lib.df[col] = tcyc_df[col]
            feat_cols.append(col)

        feat_cols = self._dedup_keep_order(feat_cols)
        keep = [c for c in feat_cols if c in lib.df.columns]
        return lib.df.loc[:, keep].copy()

    def _build_features_from_txt(self, dataframe: pd.DataFrame, feat_txt_path: str | Path) -> pd.DataFrame:
        """Build features from a txt feature plan.

        Args:
            dataframe: Input normalized dataframe.
            feat_txt_path: Txt file path.
        Returns:
            Feature dataframe.
        """
        lib = FeatureLibPTA(dataframe.copy())
        feat_df, skipped = lib.compute_from_txt(
            txt_path=feat_txt_path,
            strict=self.strict_txt,
            fillna_value=np.nan,
            dtype="float32",
        )
        self.last_skipped = skipped
        return feat_df

    @staticmethod
    def _apply_shift(feat_df: pd.DataFrame, shift_bars: int = 1) -> pd.DataFrame:
        """Apply bar shift on feature dataframe.

        Args:
            feat_df: Feature dataframe.
            shift_bars: Number of bars to shift.
        Returns:
            Shifted feature dataframe.
        """
        if int(shift_bars) == 0:
            return feat_df.copy()
        return feat_df.shift(int(shift_bars))

    @staticmethod
    def _fill_nan(feat_df: pd.DataFrame, mode: str = "last") -> pd.DataFrame:
        """Fill NaN values using selected mode.

        Args:
            feat_df: Feature dataframe.
            mode: Fill mode: zero, last, or linear_interp.
        Returns:
            Filled feature dataframe.
        """
        m = str(mode).strip().lower()
        if m in {"zero", "0", "fill0"}:
            return feat_df.fillna(0.0)
        if m in {"last", "ffill", "forward"}:
            return feat_df.ffill().bfill()
        if m in {"linear_interp", "linear", "interp"}:
            return feat_df.interpolate(method="linear", axis=0).ffill().bfill()
        raise ValueError(f"Unsupported fill_nan mode: {mode}")

    def _normalization(self, feat_df: pd.DataFrame, mode: str = "none", rolling_window: int | None = None) -> pd.DataFrame:
        """Apply optional normalization on feature columns.

        Args:
            feat_df: Feature dataframe.
            mode: none or z_rolling.
            rolling_window: Rolling window size.
        Returns:
            Normalized feature dataframe.
        """
        m = str(mode).strip().lower()
        if m in {"none", ""}:
            return feat_df
        if m != "z_rolling":
            raise ValueError(f"Unsupported normalization mode: {mode}")

        if rolling_window is None:
            raise ValueError("normalization mode z_rolling requires rolling_window.")
        window = int(rolling_window)
        if window <= 0:
            raise ValueError("rolling_window must be > 0.")

        min_periods = int(self.norm_min_periods)
        if min_periods <= 0:
            min_periods = 1
        if min_periods > window:
            min_periods = window

        out = feat_df.copy()
        numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        if not numeric_cols:
            return out

        vals = out[numeric_cols].astype("float64")
        mu = vals.rolling(window=window, min_periods=min_periods).mean()
        sd = vals.rolling(window=window, min_periods=min_periods).std(ddof=0)
        sd_safe = sd.mask(sd.abs() < float(self.norm_std_floor), float(self.norm_std_floor))
        z = (vals - mu) / sd_safe
        z = z.replace([np.inf, -np.inf], np.nan)
        out[numeric_cols] = z
        return out

    @staticmethod
    def _make_time_columns(index: pd.DatetimeIndex) -> pd.DataFrame:
        """Build datetime/timestamp columns from datetime index.

        Args:
            index: Datetime index.
        Returns:
            Time dataframe with datetime and timestamp columns.
        """
        idx = pd.DatetimeIndex(index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        return pd.DataFrame(
            {
                "datetime": idx,
                "timestamp": (idx.view("int64") // 1_000_000).astype("int64"),
            },
            index=idx,
        )

    def compute(self, df_raw: pd.DataFrame | None = None, feat_plan: str | Path | None = None) -> pd.DataFrame:
        """Compute final dataframe with time columns and shifted features.

        Args:
            df_raw: Optional source dataframe.
            feat_plan: all or feature txt path.
        Returns:
            Dataframe with datetime/timestamp and feature columns.
        """
        src = self._load_input_df(df_raw)
        src = self._normalize_columns(src)
        self._validate_required_columns(src)
        src = self._to_datetime_index(src)
        src = self._coerce_required_numeric(src)

        plan = self.default_feat_plan if feat_plan is None else feat_plan
        if str(plan).strip().lower() == "all":
            feat_df = self._build_features(src)
            self.last_skipped = []
        else:
            feat_df = self._build_features_from_txt(src, Path(plan))

        feat_df = self._apply_shift(feat_df, shift_bars=self.shift_bars)
        feat_df = self._fill_nan(feat_df, mode=self.fill_nan_mode)
        feat_df = self._normalization(feat_df, mode=self.norm_mode, rolling_window=self.norm_window)
        feat_df = feat_df.astype("float32")

        time_df = self._make_time_columns(feat_df.index)
        out = pd.concat([time_df, feat_df], axis=1)
        return out


__all__ = ["FeatureComputerPTA"]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for FeatureComputerPTA runner.

    Args:
        None
    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Run PTA feature computation from yaml config.")
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("feature_selection/features_computer/cfg_pta.yaml"),
        help="Path to cfg_pta.yaml.",
    )
    parser.add_argument(
        "--feat-plan",
        type=str,
        default=None,
        help="Override feat plan: 'all' or txt path.",
    )
    return parser.parse_args()

# ============= Genearting data ====================
def main() -> None:
    """Run feature computation using cfg_pta.yaml and write feature csv.

    Args:
        None
    Returns:
        None
    """
    args = _parse_args()
    cfg_path = Path(args.cfg).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError("Config root must be a mapping object.")

    fc = FeatureComputerPTA(cfg)
    feat_df = fc.compute(feat_plan=args.feat_plan)

    out_cfg = dict(cfg.get("output", {}) or {})
    out_path_raw = out_cfg.get("feat_path", "data/derived/ohlcv_fng_15m_feat.csv")
    out_path = Path(out_path_raw).expanduser()
    if not out_path.is_absolute():
        out_path = (cfg_path.parent.parent.parent / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(out_path, index=False)
    pq_path = out_path.with_suffix(".parquet")
    feat_df.to_parquet(pq_path, index=False, engine="pyarrow")

    print(f"[OK] Config: {cfg_path}")
    print(f"[OK] Input : {cfg.get('data', {}).get('ohlcv_fng_path', '')}")
    print(f"[OK] CSV   : {out_path}")
    print(f"[OK] Parquet: {pq_path}")
    print(f"[OK] Shape : {feat_df.shape}")
    if fc.last_skipped:
        print(f"[WARN] skipped features ({len(fc.last_skipped)}): {fc.last_skipped}")


if __name__ == "__main__":
    main()
wd