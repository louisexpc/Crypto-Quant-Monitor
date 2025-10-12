# train/evaluation/utils.py

import yaml
import copy, json, os
from pathlib import Path
import optuna
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,fbeta_score, 
)
from sklearn.preprocessing import label_binarize
from train.evaluation.reporters.classification_reporter import ClassificationReporter
from train.evaluation.reporters.regression_reporter import RegressionReporter
from train.evaluation.exporters.cv_summary import save_cv_summary

# 供 XGB 分支使用：與 DL 版混合目標對齊
def compute_mixed_objective_np(y_true, y_pred, *, alpha: float, beta: float, ema_decay: float):
    """
    Wrap metrics_reg.mixed_objective for numpy arrays.
    Returns (objective_value, components_dict).
    """
    from train.training.metrics.metrics_reg import mixed_objective
    val, comps = mixed_objective(y_true, y_pred, alpha=alpha, beta=beta, ema_decay=ema_decay)
    return val, comps

# =========================
# 公用小工具
# =========================
def _ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _ensure_2d_prob(y_prob: np.ndarray) -> np.ndarray:
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, 1)
    return y_prob

# =========================
# 訓練過程輸出（分類/回歸皆相容）
# =========================
def save_fold_metrics(metrics: list[dict], save_dir: Path, prefix: str = "",
                      y_true=None, y_pred=None, class_names=None):
    """
    儲存單個 fold 的訓練曲線。
    會自動偵測欄位並繪圖：
      - Loss（必要）
      - Acc / F1（若存在）
      - Pearson / RMSE / MAE（若存在）
      - 混淆矩陣（若提供 y_true/y_pred 與 class_names）
    """
    save_dir = _ensure_dir(save_dir)
    df = pd.DataFrame(metrics)
    df.to_csv(save_dir / f"{prefix}metrics_epoch.csv", index=False)

    # ----- Loss curve -----
    plt.figure()
    if "train_loss" in df.columns: plt.plot(df["train_loss"], label="Train Loss")
    if "val_loss" in df.columns:   plt.plot(df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}loss_curve.png"); plt.close()

    # ----- Classification curves (if exists) -----
    if "train_acc" in df.columns or "val_acc" in df.columns:
        plt.figure()
        if "train_acc" in df.columns: plt.plot(df["train_acc"], label="Train Acc")
        if "val_acc" in df.columns:   plt.plot(df["val_acc"], label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.legend(); plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}accuracy_curve.png"); plt.close()

    if "train_macro_f1" in df.columns or "val_macro_f1" in df.columns:
        plt.figure()
        if "train_macro_f1" in df.columns: plt.plot(df["train_macro_f1"], label="Train F1")
        if "val_macro_f1" in df.columns:   plt.plot(df["val_macro_f1"], label="Val F1")
        plt.xlabel("Epoch"); plt.ylabel("F1")
        plt.legend(); plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}f1_curve.png"); plt.close()

    # ----- Regression curves (if exists) -----
    # 支援欄位：train_pearson/val_pearson, train_rmse/val_rmse, train_mae/val_mae
    for metric_name, ylabel in [
        ("pearson", "Pearson r"),
        ("spearman", "Spearman ρ"),
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("mse", "MSE"),
    ]:
        tr_col, va_col = f"train_{metric_name}", f"val_{metric_name}"
        if tr_col in df.columns or va_col in df.columns:
            plt.figure()
            if tr_col in df.columns: plt.plot(df[tr_col], label=f"Train {metric_name.upper()}")
            if va_col in df.columns: plt.plot(df[va_col], label=f"Val {metric_name.upper()}")
            plt.xlabel("Epoch"); plt.ylabel(ylabel)
            plt.legend(); plt.tight_layout()
            plt.savefig(save_dir / f"{prefix}{metric_name}_curve.png"); plt.close()

    # ----- 混淆矩陣（如有提供） -----
    if y_true is not None and y_pred is not None and class_names is not None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}confusion_matrix.png")
        plt.close()

# =========================
# 兼容性 Wrapper（讓舊程式可直接用）
# =========================
def plot_test_eval(y_true, y_pred, y_prob, save_dir, prefix="", class_names=None, threshold: float | None = None):
    """
    舊版 API 的向下相容包裝：分類圖表。
    """
    reporter = ClassificationReporter(save_dir=save_dir, prefix=prefix, class_names=class_names)
    return reporter.plot_eval(y_true=y_true, y_pred=y_pred, y_prob=y_prob, threshold=threshold)

def plot_regression_eval(y_true, y_pred, save_dir, prefix=""):
    reporter = RegressionReporter(save_dir=save_dir, prefix=prefix)
    return reporter.plot_eval(y_true=y_true, y_pred=y_pred)

def plot_regression_threshold_sweep(y_true_reg, y_pred_reg, true_threshold=0.0, beta=0.5, grid_points=101, save_dir=".", prefix=""):
    reporter = RegressionReporter(save_dir=save_dir, prefix=prefix)
    return reporter.threshold_sweep(
        y_true_reg=y_true_reg, y_pred_reg=y_pred_reg,
        true_threshold=true_threshold, beta=beta, grid_points=grid_points
    )
