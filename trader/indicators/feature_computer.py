from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Sequence

from indicators.indicators_lib import IndicatorLibrary


class FeatureComputer:
    """
    1. 說明:
        單純的特徵計算器：吃原始 OHLCV(+FNG/Trades) DataFrame，依配置計算指標、shift(1)、可選 rolling z-score，輸出特徵 DataFrame。
    2. 特性:
        - 不做檔案 I/O，呼叫端負責讀/寫。
        - 僅檢查欄位/時間頻率，不補齊缺口。
        - normalization 只支援 rolling z-score，內嵌於本檔。
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        1. 說明:
            初始化計算器，讀取基本設定與多空特徵配置路徑。
        2. inputs:
            - cfg: dict，需含 compute_config.yaml 中的鍵。
        3. return:
            - None
        """
        self.cfg = cfg
        self.time_cfg = cfg.get("time", {}) or {}
        self.ohlcv_required = cfg.get("ohlcv_required", []) or []
        self.trades_cfg = cfg.get("trades", {}) or {}
        self.feat_plan_cfg = cfg.get("feat_plan", {}) or {}
        self.norm_cfg = cfg.get("feat_normalization", {}) or {}
        self.nan_policy = str(cfg.get("nan_policy", "raise")).lower()
        self.manifest_path = cfg.get("manifest_path")
        self.selected_feat_path = (cfg.get("selected_feat_path") or {})
        self._whitelist_cache: Dict[str, set[str]] = {}
        if not self.feat_plan_cfg.get("long_feat_path") or not self.feat_plan_cfg.get("short_feat_path"):
            raise ValueError("feat_plan.long_feat_path / short_feat_path are required.")

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------
    def compute(self, df_raw: pd.DataFrame, side: str = "long") -> pd.DataFrame:
        """
        1. 說明:
            接收原始 OHLCV(+FNG/Trades) df，依 side 計畫計算指標並回傳特徵 df。
        2. inputs:
            - df_raw: 原始 df，需含時間欄或 DatetimeIndex
            - side: "long" | "short"
        3. return:
            - pd.DataFrame：時間索引為 UTC，僅含計算後的特徵欄位（不含原 OHLCV）
        """
        side = str(side).lower()
        if side not in {"long", "short"}:
            raise ValueError("side must be 'long' or 'short'.")

        df = self._normalize_time_index(df_raw, self.time_cfg.get("columns", ["datetime", "timestamp"]))
        self._validate_freq(df.index, self.time_cfg.get("freq", "15min"))
        self._validate_ohlcv(df, self.ohlcv_required)
        df = self._apply_nan_policy(df)

        feat_specs = self._load_feat_plan(side)
        if not feat_specs:
            raise ValueError(f"{side} feature plan has no enabled features.")

        whitelist = self._load_whitelist(side)
        if whitelist:
            feat_specs = self._filter_feat_list_by_whitelist(feat_specs, whitelist, df)

        self.lib = IndicatorLibrary(df)
        parts: List[pd.DataFrame] = []
        for spec in feat_specs:
            name = spec["name"]
            kwargs = spec.get("kwargs", {}) or {}
            parts.append(self._build_one(name, kwargs))

        feat_df = self._finalize(parts)
        feat_df = self._apply_nan_policy(feat_df)
        feat_df = self._apply_rolling_zscore(feat_df, self.norm_cfg)
        if whitelist:
            keep_cols = [c for c in feat_df.columns if c in whitelist]
            if not keep_cols:
                raise ValueError(
                    f"No columns matched whitelist for side={side}; whitelist size={len(whitelist)}"
                )
            feat_df = feat_df.loc[:, keep_cols]
        return feat_df

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _load_feat_plan(self, side: str) -> List[Dict[str, Any]]:
        key = f"{side}_feat_path"
        path = self.feat_plan_cfg.get(key, None)
        if not path:
            raise ValueError(f"feat_plan.{key} is required for side={side}.")
        with Path(path).open("r", encoding="utf-8") as f:
            plan = yaml.safe_load(f) or {}
        feats = plan.get("features") or []
        # 支援 nested 結構: {"features": {"plan": {"features": [...]}}}
        if isinstance(feats, dict):
            feats = (feats.get("plan") or {}).get("features", [])
        normalized: List[Dict[str, Any]] = []
        for item in feats:
            spec = self._normalize_feature_spec(item)
            if spec.get("enabled", False):
                normalized.append(spec)
        return normalized

    @staticmethod
    def _normalize_feature_spec(item: Dict[str, Any]) -> Dict[str, Any]:
        """將單一特徵規格標準化為 {name, enabled, kwargs} 格式。"""
        if "name" in item:
            spec = dict(item)
        else:
            reserved = {"enabled", "kwargs"}
            builder_keys = [k for k in item.keys() if k not in reserved]
            if not builder_keys:
                raise ValueError(f"特徵規格缺少 name: {item}")
            builder = builder_keys[0]
            value = item.get(builder)
            spec = {k: v for k, v in item.items() if k != builder}
            spec["name"] = builder

            if value is not None:
                if not isinstance(value, dict):
                    raise ValueError(f"特徵規格格式錯誤（需為 dict）: {item}")
                nested = dict(value)
                nested_kwargs = dict(nested.pop("kwargs", {}) or {})
                spec_kwargs = dict(spec.get("kwargs", {}) or {})
                spec_kwargs.update(nested_kwargs)
                for key, val in nested.items():
                    if key in {"enabled"}:
                        spec[key] = val
                    else:
                        spec_kwargs[key] = val
                spec["kwargs"] = spec_kwargs

        kwargs = dict(spec.get("kwargs", {}) or {})
        extra_keys = [k for k in list(spec.keys()) if k not in {"name", "enabled", "kwargs"}]
        for key in extra_keys:
            kwargs[key] = spec.pop(key)
        spec["kwargs"] = kwargs
        spec.setdefault("enabled", True)
        return spec

    def _build_one(self, name: str, kwargs: Dict[str, Any]) -> pd.DataFrame:
        key = str(name).upper()
        if key not in self.lib.builders:
            raise ValueError(f"Unknown indicator: {key}")
        df = self.lib.builders[key](kwargs or {})
        df = df.shift(1)
        return df

    @staticmethod
    def _finalize(parts: List[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame()
        X = pd.concat(parts, axis=1)
        X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
        dups = X.columns[X.columns.duplicated()]
        if len(dups):
            raise ValueError(f"Duplicate feature names: {list(dups)}")
        return X

    # ----------------------------
    # time/index helpers
    # ----------------------------
    @staticmethod
    def _to_datetime(col: pd.Series) -> pd.DatetimeIndex:
        numeric = pd.to_numeric(col, errors="coerce")
        if numeric.notna().sum() >= len(col) * 0.5:
            sample = numeric.dropna()
            unit = "ms" if (not sample.empty and sample.iloc[0] >= 1_000_000_000_000) else "s"
            idx = pd.to_datetime(numeric, unit=unit, utc=True)
        else:
            idx = pd.to_datetime(col, utc=True)
        return pd.DatetimeIndex(idx)

    def _normalize_time_index(self, df_raw: pd.DataFrame, time_cols: Sequence[str]) -> pd.DataFrame:
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
        if not freq:
            return
        diffs = idx.to_series().diff().dropna().unique()
        if len(diffs) != 1 or diffs[0] != pd.Timedelta(freq):
            raise ValueError(f"index is not contiguous with freq={freq}.")

    def _validate_ohlcv(self, df: pd.DataFrame, required_cols: Sequence[str]) -> None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"缺少欄位: {missing}")
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    def _apply_nan_policy(self, df: pd.DataFrame) -> pd.DataFrame:
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
        if not norm_cfg.get("enabled", False):
            return df
        window = norm_cfg.get("rolling_window", None)
        if window is None:
            return df
        skip = norm_cfg.get("skip_cols", []) or []
        numeric_cols = [c for c in df.columns if c not in skip and np.issubdtype(df[c].dtype, np.number)]
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

    # ----------------------------
    # whitelist helpers
    # ----------------------------
    def _load_whitelist(self, side: str) -> set[str] | None:
        """依 side 讀取白名單檔案；若未設定則回傳 None。"""
        if side in self._whitelist_cache:
            return self._whitelist_cache[side]
        path = None
        if isinstance(self.selected_feat_path, dict):
            path = self.selected_feat_path.get(side)
        if not path:
            return None
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Whitelist file not found for side={side}: {path}")
        with path.open("r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        whitelist = set(names)
        if not whitelist:
            raise ValueError(f"Whitelist file is empty for side={side}: {path}")
        self._whitelist_cache[side] = whitelist
        return whitelist

    def _filter_feat_list_by_whitelist(
        self,
        feat_list: List[Dict[str, Any]],
        whitelist: set[str],
        base_df: pd.DataFrame,
        probe_rows: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        僅保留可能產出白名單欄位的 builder，避免交易時計算不必要的特徵。
        """
        if not whitelist:
            return feat_list

        sample_df = base_df.head(probe_rows)
        sample_lib = IndicatorLibrary(sample_df)

        kept: List[Dict[str, Any]] = []
        for item in feat_list:
            name = str(item.get("name"))
            kwargs = item.get("kwargs", {}) or {}
            try:
                key = name.upper()
                if key not in sample_lib.builders:
                    kept.append(item)
                    continue
                df_sample = sample_lib.builders[key](kwargs)
                cols = list(df_sample.columns)
            except Exception:
                kept.append(item)
                continue
            if any(c in whitelist for c in cols):
                kept.append(item)

        if not kept:
            raise ValueError("Whitelist 過濾後沒有可計算的特徵，請檢查白名單或計畫設定。")
        return kept


__all__ = ["FeatureComputer"]
