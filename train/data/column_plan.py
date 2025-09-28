# train/data/column_plan.py
"""Feature column selection helpers.

目前 pipeline 改為「預算特徵檔提供哪些欄位就使用哪些欄位」，不再依賴
`cfg.features.plan` 或分鐘白名單逐一設定。`select_plan_columns` 會回傳資料
中所有數值欄位（保留原始順序），以供 dataloader 直接使用。
"""

from __future__ import annotations

from typing import List, Dict, Optional
import pandas as pd
from pandas.api.types import is_numeric_dtype

__all__ = ["select_plan_columns"]


def select_plan_columns(feat_df: pd.DataFrame, cfg: Optional[Dict] = None) -> List[str]:
    """Return all numeric feature columns from the precomputed dataframe.

    參數
    ----
    feat_df : pd.DataFrame
        離線預算好的特徵表。函式會挑出 dtype 屬於數值型的欄位並維持原順序。
    cfg : dict | None
        保留舊介面，相容既有呼叫；目前不使用。

    回傳
    ----
    List[str]
        可用於訓練的欄位名稱清單（僅包含數值欄）。若沒有數值欄位則拋出錯誤。
    """
    numeric_cols = [col for col in feat_df.columns if is_numeric_dtype(feat_df[col])]
    if not numeric_cols:
        raise ValueError("預算特徵資料中找不到數值欄位，無法建立訓練特徵。")
    return list(numeric_cols)
