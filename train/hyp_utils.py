# hyp_utils.py

import numpy as np
import torch
import random
import re
import pandas as pd
import optuna


# ---------------------------
# 工具：設定隨機種子
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------
# 特徵過濾與 shift
# ---------------------------
def build_feature_pool(df_cols: list, cfg: dict) -> list:
    """
    根據 cfg 中的 families 過濾特徵欄位（排除 reserved）。
    """
    reserved = set(cfg["data"]["columns"]["time"] + cfg["data"]["columns"]["ohlcv"] + ["label"])
    cols = [c for c in df_cols if c not in reserved]

    fams = cfg['features'].get('families', {})
    if not fams:
        return cols

    patterns = []
    for pats in fams.values():
        patterns.extend(pats)

    pat = re.compile("|".join(patterns))
    pool = [c for c in cols if pat.search(c)]
    return pool if pool else cols


# ---------------------------
# 特徵子集抽樣器（K-subset）
# ---------------------------
def sample_k_subset(
    trial: optuna.trial.Trial,
    pool: list,
    always_on: list,
    k_range=(64, 256)
) -> list:
    """
    從 pool 中抽出 k 個特徵（保留 always_on），由 trial 控制 k 值與隨機種子。
    """
    k = trial.suggest_int("k_features", k_range[0], k_range[1])
    seed = trial.suggest_int("feat_seed", 0, 10**6 - 1)
    rng = np.random.default_rng(seed)

    base = [f for f in always_on if f in pool]
    rest = [f for f in pool if f not in base]
    need = max(0, k - len(base))
    take = rng.choice(rest, size=min(need, len(rest)), replace=False).tolist()

    return base + take
