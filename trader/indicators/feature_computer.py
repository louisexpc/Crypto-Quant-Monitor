from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Sequence

from feature_selection.features_computer.feat_lib_pta import FeatureLibPTA


class FeatureComputer:
    """Lightweight feature computer for trading runtime.

    Wraps ``FeatureLibPTA`` to compute features from a txt-based
    feature list, applies shift(1) to prevent lookahead, and
    optionally normalizes via rolling z-score.

    Args:
        cfg: Config dict (see ``feature.yaml``).
    Returns:
        None.
    """

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize with config dict.

        Args:
            cfg: Feature config containing selected_feat_path, time, normalization, etc.
        Returns:
            None.
        """
        self.cfg = cfg
        self.time_cfg = cfg.get("time", {}) or {}
        self.ohlcv_required = cfg.get("ohlcv_required", []) or []
        self.norm_cfg = cfg.get("feat_normalization", {}) or {}
        self.nan_policy = str(cfg.get("nan_policy", "raise")).lower()

        selected = cfg.get("selected_feat_path") or {}
        if not selected.get("long") and not selected.get("short"):
            raise ValueError("selected_feat_path.long or .short must be set.")
        self._txt_paths: Dict[str, str] = {}
        for side in ("long", "short"):
            p = selected.get(side)
            if p:
                self._txt_paths[side] = str(p)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(self, df_raw: pd.DataFrame, side: str = "long") -> pd.DataFrame:
        """Compute features for the given side.

        Args:
            df_raw: Raw OHLCV(+FNG) dataframe with time column or DatetimeIndex.
            side: ``"long"`` or ``"short"``.
        Returns:
            Feature dataframe (UTC DatetimeIndex, float32, no NaN).
        """
        side = str(side).lower()
        if side not in self._txt_paths:
            raise ValueError(
                f"No txt path configured for side='{side}'. "
                f"Available: {list(self._txt_paths.keys())}"
            )
        txt_path = self._txt_paths[side]

        # 1. Normalize time index
        df = self._normalize_time_index(
            df_raw, self.time_cfg.get("columns", ["datetime", "timestamp"])
        )
        # 2. Validate OHLCV
        self._validate_ohlcv(df, self.ohlcv_required)
        # 3. Apply NaN policy on raw input
        df = self._apply_nan_policy(df)

        # 4. Compute features via FeatureLibPTA
        lib = FeatureLibPTA(df)
        feat_df, skipped = lib.compute_from_txt(
            txt_path, strict=False, fillna_value=None, dtype=None,
        )
        if skipped:
            print(f"[FeatureComputer] WARNING: skipped unknown features: {skipped}")

        # 5. Shift to prevent lookahead
        feat_df = feat_df.shift(1)

        # 6. Clean up infinities, NaN policy, and dtype
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        feat_df = self._apply_nan_policy(feat_df)
        feat_df = feat_df.astype("float32")

        # 7. Rolling z-score normalization
        feat_df = self._apply_rolling_zscore(feat_df, self.norm_cfg)

        return feat_df

    # ------------------------------------------------------------------
    # Time / index helpers (preserved from original)
    # ------------------------------------------------------------------
    @staticmethod
    def _to_datetime(col: pd.Series) -> pd.DatetimeIndex:
        """Convert a column to DatetimeIndex.

        Args:
            col: Time column (numeric timestamp or datetime string).
        Returns:
            DatetimeIndex (UTC).
        """
        numeric = pd.to_numeric(col, errors="coerce")
        if numeric.notna().sum() >= len(col) * 0.5:
            sample = numeric.dropna()
            unit = "ms" if (not sample.empty and sample.iloc[0] >= 1_000_000_000_000) else "s"
            idx = pd.to_datetime(numeric, unit=unit, utc=True)
        else:
            idx = pd.to_datetime(col, utc=True)
        return pd.DatetimeIndex(idx)

    def _normalize_time_index(self, df_raw: pd.DataFrame, time_cols: Sequence[str]) -> pd.DataFrame:
        """Normalize time columns into UTC DatetimeIndex.

        Args:
            df_raw: Raw input dataframe.
            time_cols: Candidate time column names.
        Returns:
            Dataframe with UTC DatetimeIndex, sorted, deduplicated.
        """
        df = df_raw.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        idx = None
        for c in time_cols:
            if c in df.columns:
                idx = self._to_datetime(df[c])
                break
        if idx is None:
            if isinstance(df.index, pd.DatetimeIndex):
                idx = pd.DatetimeIndex(df.index)
            else:
                raise ValueError(f"需要時間欄位 {time_cols} 或 DatetimeIndex")
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        df.index = idx
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        for c in time_cols:
            if c in df.columns:
                df = df.drop(columns=[c])
        return df

    @staticmethod
    def _validate_freq(idx: pd.DatetimeIndex, freq: str) -> None:
        """Validate that the index is equally spaced.

        Args:
            idx: DatetimeIndex to validate.
            freq: Expected frequency string.
        Returns:
            None.
        """
        if not freq:
            return
        diffs = idx.to_series().diff().dropna().unique()
        if len(diffs) != 1 or diffs[0] != pd.Timedelta(freq):
            raise ValueError(f"index is not contiguous with freq={freq}.")

    def _validate_ohlcv(self, df: pd.DataFrame, required_cols: Sequence[str]) -> None:
        """Validate required columns exist and are numeric.

        Args:
            df: Dataframe to validate.
            required_cols: Required column names.
        Returns:
            None.
        """
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"缺少欄位: {missing}")
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # ------------------------------------------------------------------
    # NaN / normalization
    # ------------------------------------------------------------------
    def _apply_nan_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply NaN handling policy.

        Args:
            df: Input dataframe.
        Returns:
            Cleaned dataframe.
        """
        if self.nan_policy == "raise":
            if df.isna().any().any():
                raise ValueError("Input dataframe contains NaN/NaT; please sanitize upstream.")
            return df
        if self.nan_policy == "drop":
            return df.dropna()
        if self.nan_policy == "linear_interp":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols):
                df[numeric_cols] = (
                    df[numeric_cols]
                    .interpolate(method="linear", limit_direction="both")
                    .ffill()
                    .bfill()
                )
            return df
        raise ValueError(f"[nan_policy] unknown strategy: {self.nan_policy}")

    def _apply_rolling_zscore(self, df: pd.DataFrame, norm_cfg: Dict[str, Any]) -> pd.DataFrame:
        """Apply rolling z-score normalization to numeric columns.

        Args:
            df: Feature dataframe.
            norm_cfg: Normalization config with ``enabled``, ``rolling_window``, ``skip_cols``.
        Returns:
            Normalized dataframe.
        """
        if not norm_cfg.get("enabled", False):
            return df
        window = norm_cfg.get("rolling_window", None)
        if window is None:
            return df
        skip = set(norm_cfg.get("skip_cols", []) or [])
        numeric_cols = [
            c for c in df.columns
            if c not in skip and np.issubdtype(df[c].dtype, np.number)
        ]
        if not numeric_cols:
            return df
        eps = float(norm_cfg.get("std_floor", 1e-8))
        rolled = df.copy()
        for c in numeric_cols:
            s = df[c].astype("float32")
            mean = s.rolling(window=window, min_periods=1).mean()
            std = s.rolling(window=window, min_periods=1).std()
            std_safe = std.replace(0, np.nan).fillna(eps)
            rolled[c] = (s - mean) / std_safe
        rolled = rolled.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return rolled


__all__ = ["FeatureComputer"]
