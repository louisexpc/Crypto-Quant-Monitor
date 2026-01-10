# feature_selection/features_computer/feature_computer.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Sequence
import shutil
import yaml
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from feature_selection.features_computer import feat_normlization as fnorm
from feature_selection.features_computer.indicators import IndicatorLibrary


class FeatureComputer:
    """
    1. 負責載入/正規化資料（OHLCV+FNG、trades）、呼叫 IndicatorLibrary 計算特徵。
    2. 最終輸出：時間欄位 + 未 shift 的原始 OHLCV + 已 shift 的特徵。
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.data_cfg = cfg.get("data", {}) or {}
        self.export_cfg = cfg.get("export", {}) or {}
        self.plan = (cfg.get("features", {}) or {}).get("plan", {}) or {}
        self.lib: IndicatorLibrary | None = None

    # ------------------------------------------------------------
    # 輔助：feature plan
    # ------------------------------------------------------------
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

    @classmethod
    def _enabled_features(cls, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        feats = plan.get("features") or []
        normalized: List[Dict[str, Any]] = []
        for item in feats:
            spec = cls._normalize_feature_spec(item)
            if spec.get("enabled", False):
                normalized.append(spec)
        return normalized

    def _build_one(self, name: str, kwargs: Dict[str, Any]) -> pd.DataFrame:
        key = str(name).upper()
        if not self.lib or key not in self.lib.builders:
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

    # ------------------------------------------------------------
    # 輔助：rolling z-normalization
    # ------------------------------------------------------------
    def _apply_rolling_norm(self, df: pd.DataFrame) -> pd.DataFrame:
        norm_cfg = (self.export_cfg.get("feat_normlization", {}) or {})
        window = norm_cfg.get("rolling_window", None)
        if window is None:
            return df

        skip_cols: Sequence[str] = (
            "datetime",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        )
        return fnorm.rolling_zscore_dataframe(
            df,
            window=int(window),
            warmup=None,
            skip_columns=skip_cols,
        )

    # ------------------------------------------------------------
    # 輔助：讀檔與時間索引正規化
    # ------------------------------------------------------------
    @staticmethod
    def _to_datetime(col: pd.Series) -> pd.DatetimeIndex:
        """盡量判斷秒/毫秒 timestamp，否則走通用 to_datetime；統一 UTC。"""
        numeric = pd.to_numeric(col, errors="coerce")
        if numeric.notna().sum() >= len(col) * 0.5:
            sample = numeric.dropna()
            unit = "ms"
            if not sample.empty and sample.iloc[0] < 1_000_000_000_000:
                unit = "s"
            idx = pd.to_datetime(numeric, unit=unit, utc=True)
        else:
            idx = pd.to_datetime(col, utc=True)
        return pd.DatetimeIndex(idx)

    def _normalize_time_index(self, df_raw: pd.DataFrame, time_cols: List[str]) -> pd.DataFrame:
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

    def _normalize_ohlcv_fng(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        time_cols = list((self.data_cfg.get("columns", {}) or {}).get("time", ["datetime", "timestamp"]))
        df = self._normalize_time_index(df_raw, time_cols)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"缺少欄位 {col} 於 ohlcv_fng 資料")
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        return df

    def _normalize_trades(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        time_cols = list((self.data_cfg.get("columns", {}) or {}).get("time", ["datetime", "timestamp"]))
        df = self._normalize_time_index(df_raw, time_cols)
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        return df

    def _load_base_df(
        self,
        df_raw_ohlcv: pd.DataFrame | None = None,
        df_raw_trades: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        # OHLCV + FNG
        if df_raw_ohlcv is None:
            ohlcv_path = Path(self.data_cfg["ohlcv_fng_path"])
            df_main = self._normalize_ohlcv_fng(pd.read_csv(ohlcv_path))
        else:
            df_main = self._normalize_ohlcv_fng(df_raw_ohlcv)

        # Trades (optional)
        trades_cfg = self.data_cfg.get("trades", {}) or {}
        trades_norm = None
        if trades_cfg.get("enabled", False):
            trades_raw = df_raw_trades
            if trades_raw is None:
                trades_path = Path(trades_cfg["trades_min_path"])
                trades_raw = pd.read_csv(trades_path)
            trades_norm = self._normalize_trades(trades_raw)
            df_main = df_main.join(trades_norm, how="left")
        return df_main, trades_norm

    # ------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------
    def compute(
        self,
        df_raw_ohlcv: pd.DataFrame | None = None,
        df_raw_trades: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        回傳 (15m 特徵表, trades 特徵表|None)。
        若呼叫者提供 df_raw_ohlcv/df_raw_trades 則使用之，否則依 cfg 讀檔。
        """
        base_df, trades_norm = self._load_base_df(df_raw_ohlcv, df_raw_trades)
        self.lib = IndicatorLibrary(base_df)

        feat_list = self._enabled_features(self.plan)
        if not feat_list:
            raise ValueError("計畫沒有任何 enabled=True 的 features。")

        parts: List[pd.DataFrame] = []
        for item in tqdm(feat_list, desc="Building features", total=len(feat_list)):
            name = str(item.get("name"))
            kwargs = item.get("kwargs", {}) or {}
            parts.append(self._build_one(name, kwargs))

        feat_df = self._finalize(parts)

        nan_policy = str(self.export_cfg.get("nan_policy", "none")).strip().lower()
        if nan_policy == "linear_interp":
            numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols):
                feat_df[numeric_cols] = (
                    feat_df[numeric_cols]
                    .interpolate(method="linear", limit_direction="both")
                    .ffill()
                    .bfill()
                )
        elif nan_policy not in {"none", ""}:
            raise ValueError(f"[nan_policy] 未知策略: {nan_policy}")
        feat_df = feat_df.astype("float32")

        # 時間欄 + 未 shift 原始 OHLCV + 特徵
        idx = pd.DatetimeIndex(feat_df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        utc_idx = idx.tz_convert("UTC")
        time_df = pd.DataFrame(
            {
                "datetime": utc_idx,
                "timestamp": (utc_idx.view("int64") // 1_000_000).astype("int64"),  # ms
            },
            index=feat_df.index,
        )
        raw_ohlcv = base_df.loc[feat_df.index, ["open", "high", "low", "close", "volume"]].astype("float32")

        df_ohlcv_feat = pd.concat(
            [
                time_df.reset_index(drop=True),
                raw_ohlcv.reset_index(drop=True),
                feat_df.reset_index(drop=True),
            ],
            axis=1,
        )
        df_ohlcv_feat = self._apply_rolling_norm(df_ohlcv_feat)

        df_trades_feat = None
        if trades_norm is not None:
            # 只保留設定指定的 1m 欄位（如 min_trade_feat）；未設定則保留全部。
            keep_cols = (self.export_cfg.get("min_trade_feat") or None)
            if keep_cols:
                keep_cols = [c for c in keep_cols if c in trades_norm.columns]
                trades_norm = trades_norm.loc[:, keep_cols]

            idx_t = pd.DatetimeIndex(trades_norm.index)
            if idx_t.tz is None:
                idx_t = idx_t.tz_localize("UTC")
            time_df_t = pd.DataFrame(
                {
                    "datetime": idx_t,
                    "timestamp": (idx_t.view("int64") // 1_000_000).astype("int64"),
                },
                index=trades_norm.index,
            )
            trades_out_df = pd.concat(
                [time_df_t.reset_index(drop=True), trades_norm.reset_index(drop=True)],
                axis=1,
            )
            norm_cfg = (self.export_cfg.get("feat_normlization", {}) or {})
            t_window = norm_cfg.get("rolling_window", None)
            if t_window is None:
                df_trades_feat = trades_out_df
            else:
                df_trades_feat = fnorm.rolling_zscore_dataframe(
                    trades_out_df,
                    window=int(t_window),
                    warmup=None,
                    skip_columns=("datetime", "timestamp"),
                )
        return df_ohlcv_feat, df_trades_feat


def main():
    cfg_path = Path("feature_selection/features_computer/features_config.yaml")
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fc = FeatureComputer(cfg)
    ohlcv_df, trades_df = fc.compute()

    out_dir = Path(fc.export_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ohlcv_path = out_dir / fc.export_cfg["ohlcv_feat"]
    if ohlcv_path.suffix.lower() == ".parquet":
        ohlcv_df.to_parquet(ohlcv_path, index=False)
    else:
        ohlcv_df.to_csv(ohlcv_path, index=False)

    trades_cfg = fc.data_cfg.get("trades", {}) or {}
    trades_out = None
    if trades_cfg.get("enabled", False) and trades_df is not None:
        trades_out = out_dir / fc.export_cfg["trades_feat"]
        trades_out.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(trades_out, index=False)

    cfg_copy_path = out_dir / cfg_path.name
    shutil.copy2(cfg_path, cfg_copy_path)

    print(f"[OK] Exported ohlcv features to: {ohlcv_path}")
    if trades_out:
        print(f"[OK] Exported trades features to: {trades_out}")
    print(f"[OK] Copied config to: {cfg_copy_path}")
    

if __name__ == "__main__":
    main()
