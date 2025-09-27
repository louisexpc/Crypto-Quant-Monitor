# train/build_feature_loader/feature_store.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd

from ..build_feature_loader.build_features import create_label

@dataclass(frozen=True)
class FrameBundle:
    """
    1. 說明: 封裝已載入的特徵與（可選）標籤
    2. inputs: 無（dataclass）
    3. return: dataclass 物件
    """
    features: pd.DataFrame
    labels: Optional[pd.Series]
    columns: List[str]

class FeatureStore:
    """
    以「一次載入、多處重用」為目標的特徵快取層。
    - 僅處理: 讀檔、UTC 索引、排序/去重；不做縮放（scaler 留在 loader）
    - 可選: 依 cfg.label.* 預先生成時間驅動標籤（事件任務則略過）
    - 安全: 輸出都回傳 copy（避免外部改到內部狀態）
    """

    def __init__(self, cfg: dict):
        """
        1. 說明: 建構子；不做重活，請呼叫 from_cfg()
        2. inputs:
           - cfg: 設定 dict
        3. return: None
        """
        self._cfg = cfg
        self._bundle: Optional[FrameBundle] = None

    @staticmethod
    def _read_any(path: str) -> pd.DataFrame:
        """
        1. 說明: 讀取 CSV/Parquet 檔案
        2. inputs:
           - path: 檔案路徑
        3. return:
           - pd.DataFrame
        """
        p = str(path)
        if p.endswith(".parquet"):
            return pd.read_parquet(p)
        if p.endswith(".csv"):
            return pd.read_csv(p)
        raise ValueError("只支援 .csv / .parquet")

    @staticmethod
    def _as_sorted_utc_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        1. 說明: 將 'datetime' 或 'timestamp' 轉成 UTC DatetimeIndex、排序、去重
        2. inputs:
           - df: 原始 DataFrame
        3. return:
           - pd.DataFrame: 以 DatetimeIndex 排序好的表
        """
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
        elif "datetime" in df.columns:
            idx = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        elif "timestamp" in df.columns:
            ts = pd.to_numeric(df["timestamp"], errors="coerce")
            unit = "ms" if ts.dropna().iloc[0] > 1_000_000_000_000 else "s"
            idx = pd.to_datetime(ts, unit=unit, utc=True)
        else:
            raise ValueError("資料需有 'datetime' 或 'timestamp' 欄，或原本就是 DatetimeIndex")
        df2 = df.copy()
        df2.index = pd.DatetimeIndex(idx).sort_values()
        df2 = df2[~df2.index.duplicated(keep="last")]
        return df2

    @classmethod
    def from_cfg(cls, cfg: dict, *, compute_time_labels: bool = True) -> "FeatureStore":
        """
        1. 說明: 依 cfg 一次性建立快取
        2. inputs:
           - cfg: 專案設定
           - compute_time_labels: 是否預先計算「時間驅動」標籤（事件任務會自動忽略）
        3. return:
           - FeatureStore: 可供後續 fold 重用
        """
        inst = cls(cfg)
        data_path = cfg["data"]["path"]
        df_raw = cls._read_any(data_path)
        df_norm = cls._as_sorted_utc_index(df_raw)

        # 這裡不篩 features.plan（避免和 loader 互相踩）——讓 loader 再選欄位
        labels = None
        if cfg.get("label", {}).get("mode") != "event_tbm" and compute_time_labels:
            # 時間驅動任務：先算一次標籤，避免每個 fold 重算
            labels = create_label(df_norm, cfg, return_what="auto")

        inst._bundle = FrameBundle(features=df_norm, labels=labels, columns=list(df_norm.columns))
        return inst

    # ----------------- Public API -----------------
    def get_frame(self) -> pd.DataFrame:
        """
        1. 說明: 取得完整特徵表（已排序、UTC index）
        2. inputs: None
        3. return:
           - pd.DataFrame 的 copy（避免外部修改內部）
        """
        assert self._bundle is not None, "FeatureStore 尚未初始化"
        return self._bundle.features.copy()

    def get_labels(self) -> Optional[pd.Series]:
        """
        1. 說明: 取得預先計算的時間驅動標籤（事件任務會回 None）
        2. inputs: None
        3. return:
           - pd.Series 或 None
        """
        assert self._bundle is not None, "FeatureStore 尚未初始化"
        return None if self._bundle.labels is None else self._bundle.labels.copy()

    def get_columns(self) -> List[str]:
        """
        1. 說明: 取得原始欄位名稱（供 TwoStream 切分用）
        2. inputs: None
        3. return:
           - list[str]
        """
        assert self._bundle is not None, "FeatureStore 尚未初始化"
        return list(self._bundle.columns)
