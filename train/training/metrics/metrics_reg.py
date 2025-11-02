# metrics_reg.py
from __future__ import annotations
import numpy as np


def _rank_average(x: np.ndarray) -> np.ndarray:
    """
    以 'average' 方式處理 ties 的排名（1-based），只依賴 numpy。
    """
    x = np.asarray(x)
    n = x.size
    order = np.argsort(x, kind="mergesort")  # 穩定排序以利 ties
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        xi = x[order[i]]
        # 找到 [i..j] 的同值區間
        while j + 1 < n and x[order[j + 1]] == xi:
            j += 1
        # 平均名次（1-based）
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks

def _spearman_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman's rho = Pearson correlation of the ranked variables.
    若任一方為常數，回傳 NaN（與常見實作一致）。
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.size < 2:
        return float("nan")
    r1 = _rank_average(yt)
    r2 = _rank_average(yp)
    # Pearson of ranks
    r1c = r1 - r1.mean()
    r2c = r2 - r2.mean()
    denom = np.sqrt((r1c**2).sum()) * np.sqrt((r2c**2).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((r1c * r2c).sum() / denom)

def compute_regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true
    mse  = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(diff)))
    if y_true.size >= 2 and np.isfinite(y_true).all() and np.isfinite(y_pred).all():
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")
    spearman = _spearman_np(y_true, y_pred)
    return {"pearson": pearson, "spearman": spearman, "rmse": rmse, "mae": mae, "mse": mse}

def ema_mse(y_true, y_pred, decay: float = 0.9) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.size == 0:
        return float("nan")
    # w_t = (1-decay)*decay^{n-1-t}
    n = yt.size
    w = (1.0 - decay) * (decay ** np.arange(n - 1, -1, -1, dtype=float))
    w_sum = w.sum()
    w = w / w_sum if w_sum > 0 else np.ones_like(yt) / max(n, 1)
    return float(((yp - yt) ** 2 * w).sum())

def mixed_objective(y_true, y_pred, *, alpha: float, beta: float, ema_decay: float):
    """
    objective = alpha·EMA-MSE + beta·(1 - pearson)
    額外回傳 component：pearson、spearman、ema_mse，方便 logger 紀錄。
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ems = ema_mse(yt, yp, decay=float(ema_decay))
    if yt.size >= 2 and np.isfinite(yt).all() and np.isfinite(yp).all():
        pearson = float(np.corrcoef(yt, yp)[0, 1])
    else:
        pearson = 0.0
    spearman = _spearman_np(yt, yp)
    obj = float(alpha * ems + beta * (1.0 - pearson))
    return obj, {"pearson": pearson, "spearman": spearman, "ema_mse": ems}