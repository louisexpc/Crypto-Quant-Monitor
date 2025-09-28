# utils/metrics_cls.py
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
)


_EPS = 1e-12

def fbeta_macro(y_true, y_pred, beta: float) -> float:
    p_k, r_k, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    b2 = beta * beta
    f_k = (1.0 + b2) * (p_k * r_k) / (b2 * p_k + r_k + _EPS)
    return float(np.mean(f_k))


def compute_cls_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f05_macro = float(fbeta_score(y_true, y_pred, beta=0.5, average="macro", zero_division=0))

    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0

    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    return {
        "acc":               acc,
        "macro_precision":   float(p_macro),
        "macro_recall":      float(r_macro),
        "macro_f1":          float(f1_macro),
        "weighted_f1":       f1_weighted,
        "macro_f05":         f05_macro,     # ← 新鍵名（建議用這個）
        "f_05_macro":        f05_macro,     # ← 舊鍵名 alias（避免 KeyError）
        "mcc":               mcc,
        "prec_per_class":    prec_c.tolist(),
        "rec_per_class":     rec_c.tolist(),
        "f1_per_class":      f1_c.tolist(),
        "support_per_class": sup_c.tolist(),
    }