# train/evaluation/exporters/tbm_exporter.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class TBMExporter:
    """
    1. 說明:
        純輸出器：把 Predictor 產生的「帶預測欄位」DataFrame 寫成 CSV。
        不負責模型推論、不負責建 dataset（避免跟 Predictor 耦合）。
    2. inputs:
        - cfg: 全域設定 dict（主要用於 default 輸出路徑命名）
    3. return:
        - TBMExporter instance
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        1. 說明:
            初始化 exporter。
        2. inputs:
            - cfg: 設定 dict
        3. return:
            - None
        """
        self.cfg = cfg

    def export_csv(self, pred_df: pd.DataFrame, save_to_path: Optional[str] = None) -> str:
        """
        1. 說明:
            將 pred_df 直接輸出成 CSV（pred_df 已是完整 TBM events 表 + 預測欄位）。
        2. inputs:
            - pred_df: Predictor 輸出的 DataFrame
            - save_to_path: 輸出路徑；若 None，使用 cfg.label.tbm_csv_path 同資料夾預設命名
        3. return:
            - save_path (str)
        """
        if save_to_path is None:
            tbm_path = str((self.cfg.get("label", {}) or {}).get("tbm_csv_path"))
            out_dir = Path(tbm_path).parent
            base = "pred"
            save_to_path = str(out_dir / f"tbm_with_{base}.csv")

        Path(save_to_path).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(save_to_path, index=False)
        return save_to_path
