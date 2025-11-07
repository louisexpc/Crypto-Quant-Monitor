from __future__ import annotations
import fnmatch
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_yaml(path: str | Path) -> Dict:
    """ 1. 說明: 載入 YAML 設定檔
        2. inputs: path: 檔案路徑
        3. return: 以 dict 表示的設定內容 """
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def ensure_dir(p: str | Path) -> Path:
    """ 1. 說明: 確保輸出資料夾存在（無則建立）
        2. inputs: p: 路徑字串或 Path
        3. return: Path 物件（已存在的資料夾） """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_purged_kfold_indices(
    times: pd.DatetimeIndex, n_splits: int, embargo_minutes: int, shuffle: bool = False
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """ 1. 說明: 產生 Purged K-Fold + Embargo 的索引切分
        2. inputs:
            - times: DatetimeIndex（升冪）
            - n_splits: 幾折
            - embargo_minutes: 減少資訊外洩的緩衝分鐘數（建議 ≥ 標籤持有期）
            - shuffle: 是否打亂（金融時序建議 False）
        3. return: list of (train_idx, test_idx) """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=None)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    emb = pd.Timedelta(minutes=int(embargo_minutes))

    for train_idx, test_idx in kf.split(times):
        test_start = times[test_idx].min()
        test_end = times[test_idx].max()
        left_forbid_start = test_start - emb
        right_forbid_end = test_end + emb

        train_mask = np.ones(len(times), dtype=bool)
        forbid = (times >= left_forbid_start) & (times <= right_forbid_end)
        train_mask[forbid] = False
        train_mask[test_idx] = False

        new_train_idx = np.where(train_mask)[0]
        splits.append((new_train_idx, test_idx))
    return splits


def select_feature_columns(
    df: pd.DataFrame,
    datetime_col: str,
    exclude_patterns: List[str] | None = None,
    include_prefixes: List[str] | None = None,
    auto_exclude_labels: bool = True,
) -> List[str]:
    """ 1. 說明: 擇取要進 PCA/UMAP 的連續特徵欄（自動排除時間/標籤/旗標等）
        2. inputs:
            - df: 原始資料表
            - datetime_col: 時間欄位名稱
            - exclude_patterns: 需排除的樣式（如 ["*_flag"]）
            - include_prefixes: 僅納入這些前綴開頭的欄（None 表示不限制）
            - auto_exclude_labels: 是否自動排除所有 "y_*" 欄（標籤）
        3. return: 欄名清單（僅數值型） """
    all_cols = df.columns.tolist()
    patterns = list(exclude_patterns or [])
    if auto_exclude_labels:
        patterns += ["y_*"]

    # 先做 include（若有指定）
    if include_prefixes:
        candidates = [c for c in all_cols if any(c.startswith(p) for p in include_prefixes)]
    else:
        candidates = [c for c in all_cols if c != datetime_col]

    # 套用排除樣式
    def _excluded(col: str) -> bool:
        return any(fnmatch.fnmatch(col, pat) for pat in patterns)

    candidates = [c for c in candidates if not _excluded(c)]

    # 僅保留數值欄
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in candidates if c in numeric_cols]
    return candidates


def find_pc_columns(df: pd.DataFrame, pc_prefix: str = "PC") -> List[str]:
    """ 1. 說明: 從表格中找出 PC 欄位（PC1..PCk），並依序排序
        2. inputs: df: DataFrame, pc_prefix: 例 'PC'
        3. return: 依數字排序的 PC 欄名清單 """
    pcs: List[str] = []
    for c in df.columns:
        if c.startswith(pc_prefix):
            try:
                _ = int(c[len(pc_prefix):])
                pcs.append(c)
            except ValueError:
                pass
    pcs.sort(key=lambda x: int(x[len(pc_prefix):]))
    return pcs

