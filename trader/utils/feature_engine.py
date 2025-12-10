# feature_engine.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
import pandas as pd

# 根據實際路徑調整這行
from utils.indicators import IndicatorLibrary, FeatureComputer  # type: ignore


CfgType = Dict[str, Any]


@dataclass
class FeatureEngine:
    """
    高階特徵引擎：
    - 負責載入/保存 yaml config
    - 依 config 規範，將 OHLCV DataFrame → 特徵矩陣 X
    - 不做 I/O，不綁檔名，方便在 service 裡長期常駐
    """
    cfg: CfgType
    name: str = "default"

    # ---------- 建構 & config ----------

    @classmethod
    def from_yaml(cls, path: Union[str, Path], name: Optional[str] = None) -> "FeatureEngine":
        """
        從 yaml 檔建構引擎：
        - path: feature_108_short.yaml 的路徑
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg=cfg, name=name or path.stem)

    @property
    def _feat_cfg(self) -> CfgType:
        return self.cfg.get("features", {}) or {}

    @property
    def _plan(self) -> dict:
        """
        將 config 裡的 features.plan（通常是 list）轉成
        FeatureComputer 期待的格式：{"features": [...]}。
        若本來就是 dict，則直接沿用，保持相容。
        """
        feat_cfg = self._feat_cfg          # = cfg.get("features", {}) or {}
        plan = feat_cfg.get("plan")

        # 情況 1：YAML 像 feature_108_short/long.yaml 一樣，plan 是一串 list
        #   features:
        #     plan:
        #       - name: SMA
        #         ...
        if isinstance(plan, list):
            return {"features": plan}

        # 情況 2：之後如果改成：
        #   features:
        #     plan:
        #       features:
        #         - name: SMA ...
        # 也能相容
        if isinstance(plan, dict):
            # 如果裡面已經有 "features" key 就直接用
            if "features" in plan:
                return plan
            # 沒有就包起來，避免之後出現其他欄位
            return {"features": plan.get("features", [])}

        # 情況 3：某些配置沒有 plan，直接拋錯
        raise ValueError("cfg['features']['plan'] 格式錯誤或不存在，請檢查 feature YAML。")

    @property
    def _data_cfg(self) -> CfgType:
        return self.cfg.get("data", {}) or {}

    @property
    def freq_check(self) -> Optional[str]:
        """對應原本 cfg['data']['freq']（可為 None）。"""
        return self._data_cfg.get("freq")

    @property
    def index_col(self) -> str:
        """對應原本 cfg['data']['index_col']，預設 'timestamp'。"""
        return self._data_cfg.get("index_col", "timestamp")

    # ---------- 內部建構元件 ----------

    def _make_library(self, df_raw: pd.DataFrame) -> IndicatorLibrary:
        """
        依 config 建立一個新的 IndicatorLibrary：
        - 每次給不同 df_raw 時，都會建新的 lib（保持無狀態、thread-safe）
        """
        return IndicatorLibrary(
            df_raw=df_raw,
            freq_check=self.freq_check,
            prefer_time_col=self.index_col,
        )

    def _make_computer(self, df_raw: pd.DataFrame) -> FeatureComputer:
        """
        依當前 df_raw 準備 FeatureComputer。
        """
        lib = self._make_library(df_raw)
        return FeatureComputer(lib)

    # ---------- 對外主要 API ----------

    def compute_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        核心方法：
        - input : 原始 OHLCV DataFrame（含 open/high/low/close/volume + 時間欄）
        - output: 依 yaml plan 設定計算出的特徵 X（float32，index 為對齊後 DatetimeIndex）

        注意：
        - shift(1)、flip、dropna 等邏輯全部沿用原本 FeatureComputer.compute(...)
        - yaml 裡的 features.dropna, features.min_trade_feat 等設定仍會生效
        """
        fc = self._make_computer(df_raw)
        X = fc.compute(plan=self._plan, cfg=self.cfg, load_if_exists=False)
        return X

    def feature_names(self, df_sample: pd.DataFrame) -> list[str]:
        """
        給定一個 sample 的 OHLCV df，推估此 config 會產出的欄位名稱列表。
        - 不讀檔、不寫檔，只用 FeatureComputer.columns_for_plan(...)
        """
        fc = self._make_computer(df_sample)
        cols = fc.columns_for_plan(plan=self._plan, cfg=self.cfg)
        return list(cols)

    def passthrough_minute_columns(self, df_sample: pd.DataFrame) -> list[str]:
        """
        回傳依 config 中 min_trade_feat 設定，會被當作直通特徵的 m_* 欄位名稱。
        （只是幫你把 FeatureComputer.passthrough_columns 封裝起來）
        """
        fc = self._make_computer(df_sample)
        return fc.passthrough_columns(self.cfg)
