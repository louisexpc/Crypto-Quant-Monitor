# compute_export_metrices.py

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

# 供 XGB 分支使用：與 DL 版混合目標對齊
def compute_mixed_objective_np(y_true, y_pred, *, alpha: float, beta: float, ema_decay: float):
    """
    Wrap metrics_reg.mixed_objective for numpy arrays.
    Returns (objective_value, components_dict).
    """
    from .metrics_reg import mixed_objective
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
# 分類圖表與摘要
# =========================
class ClassificationReporter:
    """
    用途：
      - 畫 ROC、PR、混淆矩陣
      - 產出 JSON 摘要（各類別 AUC/AP、micro）
    介面：
      - plot_eval(y_true, y_pred, y_prob, threshold=None)
    """
    def __init__(self, save_dir, prefix="", class_names=None):
        self.save_dir = _ensure_dir(save_dir)
        self.prefix = prefix or ""
        self.class_names = list(class_names) if class_names is not None else None

    def plot_eval(self, y_true, y_pred, y_prob, threshold: float | None = None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_prob = _ensure_2d_prob(y_prob)

        if self.class_names is None:
            n_classes = int(np.max(y_true)) + 1
            self.class_names = [f"C{i}" for i in range(n_classes)]
        n_classes = len(self.class_names)

        # 固定類別編碼
        Y = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes == 2 and Y.shape[1] == 1:
            # 轉成 [N,2] 的 one-hot
            Y = np.concatenate([1 - Y, Y], axis=1)
        if n_classes == 2 and y_prob.shape[1] == 1:
            # 機率只有一欄 → 視為正類 p1
            p1 = y_prob[:, 0]
            y_prob = np.stack([1.0 - p1, p1], axis=1)

        # ----- ROC -----
        roc_auc = {}
        plt.figure(figsize=(6, 5))
        for c in range(n_classes):
            if Y[:, c].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(Y[:, c], y_prob[:, c])
            auc_c = auc(fpr, tpr)
            roc_auc[self.class_names[c]] = float(auc_c)
            plt.plot(fpr, tpr, label=f"{self.class_names[c]} (AUC={auc_c:.3f})")

        fpr_m, tpr_m, _ = roc_curve(Y.ravel(), y_prob.ravel())
        auc_m = auc(fpr_m, tpr_m)
        roc_auc["micro_avg"] = float(auc_m)
        plt.plot(fpr_m, tpr_m, "--", label=f"micro-avg (AUC={auc_m:.3f})")
        plt.plot([0, 1], [0, 1], ":", label="chance")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("ROC (OvR)"); plt.legend(); plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}roc_curve.png", dpi=200)
        plt.close()

        # ----- PR -----
        pr_auc = {}
        plt.figure(figsize=(6, 5))
        thr_point = None
        for c in range(n_classes):
            if Y[:, c].sum() == 0:
                continue
            prec, rec, thr = precision_recall_curve(Y[:, c], y_prob[:, c])
            ap = average_precision_score(Y[:, c], y_prob[:, c])
            pr_auc[self.class_names[c]] = float(ap)
            plt.plot(rec, prec, label=f"{self.class_names[c]} (AP={ap:.3f})")

            if n_classes == 2 and threshold is not None and c == (n_classes - 1):
                # 在 PR 曲線上標記閾值（以接近的 thr 索引）
                idx = np.argmin(np.abs(thr - threshold)) if len(thr) else None
                if idx is not None and 0 <= idx < len(prec):
                    p_thr, r_thr = prec[idx], rec[idx]
                    plt.plot(r_thr, p_thr, "o", markersize=8)
                    plt.annotate(
                        f"thr={threshold}\nP={p_thr:.2f}, R={r_thr:.2f}",
                        (r_thr, p_thr), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9,
                        arrowprops=dict(arrowstyle="->")
                    )
                    thr_point = {"precision": float(p_thr), "recall": float(r_thr)}

        # micro-average
        prec_m, rec_m, _ = precision_recall_curve(Y.ravel(), y_prob.ravel())
        ap_m = average_precision_score(Y.ravel(), y_prob.ravel())
        pr_auc["micro_avg"] = float(ap_m)
        plt.plot(rec_m, prec_m, "--", label=f"micro-avg (AP={ap_m:.3f})")

        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall (OvR)")
        plt.legend(); plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}pr_curve.png", dpi=200)
        plt.close()

        # ----- Confusion Matrix (row-normalized ratio) -----
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)), normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap="Blues", xticks_rotation=45, values_format=".2f")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}confusion_matrix.png", dpi=200)
        plt.close()

        # ----- JSON 摘要 -----
        summary = {}
        all_keys = set(roc_auc.keys()) | set(pr_auc.keys())
        for k in sorted(all_keys):
            summary[k] = {
                "roc_auc": float(roc_auc.get(k)) if k in roc_auc else None,
                "pr_auc":  float(pr_auc.get(k))  if k in pr_auc  else None,
            }
        if threshold is not None and n_classes == 2:
            summary["threshold"] = {"value": float(threshold)}
            if thr_point is not None:
                summary["threshold"].update(thr_point)

        with open(self.save_dir / f"{self.prefix}auc_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary


# =========================
# 回歸圖表與門檻掃描
# =========================
class RegressionReporter:
    """
    用途：
      - plot_eval(y_true, y_pred): 散點 + 殘差直方圖 + 殘差 vs y_pred
      - threshold_sweep(y_true_reg, y_pred_reg, ...): 回歸→分類之 Fβ vs 門檻
    """
    def __init__(self, save_dir, prefix=""):
        self.save_dir = _ensure_dir(save_dir)
        self.prefix = prefix or ""

    # @staticmethod
    # def _pearson_rmse_mae(y_true, y_pred):
    #     y_true = np.asarray(y_true).reshape(-1)
    #     y_pred = np.asarray(y_pred).reshape(-1)
    #     mask = np.isfinite(y_true) & np.isfinite(y_pred)
    #     y_true, y_pred = y_true[mask], y_pred[mask]
    #     if y_true.size == 0:
    #         return 0.0, float("nan"), float("nan"), np.array([]), np.array([])
    #     err = y_pred - y_true
    #     mse = float(np.mean(err ** 2))
    #     rmse = float(np.sqrt(mse))
    #     mae = float(np.mean(np.abs(err)))
    #     yt = y_true - y_true.mean()
    #     yp = y_pred - y_pred.mean()
    #     denom = (np.sqrt((yt**2).sum()) * np.sqrt((yp**2).sum()))
    #     pearson = float((yt * yp).sum() / denom) if denom > 1e-12 else 0.0
    #     return pearson, rmse, mae, y_true, y_pred

    @staticmethod
    def _corr_rmse_mae_spearman(y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if y_true.size == 0:
            return 0.0, float("nan"), float("nan"), float("nan"), np.array([]), np.array([])
        err = y_pred - y_true
        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))
        yt = y_true - y_true.mean()
        yp = y_pred - y_pred.mean()
        denom = (np.sqrt((yt**2).sum()) * np.sqrt((yp**2).sum()))
        pearson = float((yt * yp).sum() / denom) if denom > 1e-12 else 0.0

        # Spearman：以排名後做 Pearson
        def _rank_avg(a):
            a = np.asarray(a)
            n = a.size
            order = np.argsort(a, kind="mergesort")
            ranks = np.empty(n, dtype=float)
            i = 0
            while i < n:
                j = i
                ai = a[order[i]]
                while j + 1 < n and a[order[j + 1]] == ai:
                    j += 1
                avg_rank = 0.5 * (i + j) + 1.0
                ranks[order[i:j + 1]] = avg_rank
                i = j + 1
            return ranks

        r1 = _rank_avg(y_true)
        r2 = _rank_avg(y_pred)
        r1c = r1 - r1.mean()
        r2c = r2 - r2.mean()
        denom_s = (np.sqrt((r1c**2).sum()) * np.sqrt((r2c**2).sum()))
        spearman = float((r1c * r2c).sum() / denom_s) if denom_s > 1e-12 else float("nan")

        return pearson, rmse, mae, spearman, y_true, y_pred
    
    def plot_eval(self, y_true, y_pred):
        r, rmse, mae, spr, y_true, y_pred = self._corr_rmse_mae_spearman(y_true, y_pred)
        if y_true.size == 0:
            print("[RegressionReporter] empty inputs after masking; skip.")
            return {"pearson": 0.0, "spearman": float("nan"), "rmse": float("nan"), "mae": float("nan")}

        err = y_pred - y_true

        # (1) y_true vs y_pred
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, s=6, alpha=0.6)
        lim_min = float(min(y_true.min(), y_pred.min()))
        lim_max = float(max(y_true.max(), y_pred.max()))
        plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle='--', color='gray', linewidth=1.0)
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.axis("equal")
        plt.xlabel("y_true"); plt.ylabel("y_pred")
        plt.title(f"y_true vs y_pred\nPearson r={r:.3f} | Spearman ρ={spr:.3f} | RMSE={rmse:.4g} | MAE={mae:.4g}")
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}reg_scatter.png", dpi=200)
        plt.close()

        # (2) 殘差直方圖
        plt.figure(figsize=(6, 4))
        abs_max = float(np.abs(err).max())
        plt.hist(err, bins=50)
        plt.xlim(-abs_max, abs_max)
        plt.axvline(0.0, color="red", linestyle="--", linewidth=1.0)
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.xlabel("Residual (y_pred - y_true)"); plt.ylabel("Count")
        plt.title("Residual Histogram")
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}reg_residual_hist.png", dpi=200)
        plt.close()

        # (3) 殘差 vs y_pred
        plt.figure(figsize=(6, 4))
        plt.scatter(y_pred, err, s=6, alpha=0.6)
        plt.axhline(0.0, linewidth=1.0)
        plt.axvline(0.0, linestyle='--', color='gray', linewidth=1.0)
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.xlabel("y_pred"); plt.ylabel("Residual")
        plt.title("Residual vs y_pred")
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}reg_residual_vs_pred.png", dpi=200)
        plt.close()
        return {"pearson": r, "spearman": spr, "rmse": rmse, "mae": mae}
    

    def threshold_sweep(self, y_true_reg, y_pred_reg,
                        true_threshold: float = 0.0, beta: float = 0.5,
                        grid_points: int = 101):
        y_true_reg = np.asarray(y_true_reg).reshape(-1)
        y_pred_reg = np.asarray(y_pred_reg).reshape(-1)
        mask = np.isfinite(y_true_reg) & np.isfinite(y_pred_reg)
        y_true_reg, y_pred_reg = y_true_reg[mask], y_pred_reg[mask]

        y_true_bin = (y_true_reg >= float(true_threshold)).astype(int)
        qs = np.linspace(0, 1, int(grid_points))
        cand = np.quantile(y_pred_reg, qs)
        fvals = []
        for t in cand:
            yhat = (y_pred_reg >= t).astype(int)
            f = fbeta_score(y_true_bin, yhat, beta=beta, zero_division=0)
            fvals.append(f)

        fvals = np.asarray(fvals)
        best_idx = int(np.argmax(fvals))
        best_t, best_f = float(cand[best_idx]), float(fvals[best_idx])

        plt.figure(figsize=(6, 4))
        plt.plot(cand, fvals)
        plt.axvline(best_t, linestyle="--")
        plt.xlabel("prediction threshold on y_pred")
        plt.ylabel(f"F_{beta}")
        plt.title(f"F_{beta} vs threshold | best_t={best_t:.6g}, best_F={best_f:.3f}")
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}reg_threshold_sweep_f{beta}.png", dpi=200)
        plt.close()

        return {"best_threshold": best_t, "best_fbeta": best_f}

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
# 結果輸出（共用）
# =========================

# 摘要輸出（置頂 test 平均；逐 fold 列表）---
def _numeric_dict(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
            out[k] = float(v)
    return out

def _avg_std_dict(rows: list[dict]):
    pool = {}
    for d in rows:
        for k, v in _numeric_dict(d).items():
            pool.setdefault(k, []).append(v)
    avg = {k: float(np.mean(vs)) for k, vs in pool.items()}
    std = {k: float(np.std(vs, ddof=0)) for k, vs in pool.items()}
    return avg, std

def save_cv_summary(fold_results: list[dict], export_dir: str | Path, task_type: str):
    export_dir = _ensure_dir(export_dir)
    folds_out = []

    # 收集每個 fold 的 val/test 指標（分類 vs 回歸各自的鍵）
    for i, res in enumerate(fold_results):
        if task_type == "classification":
            val  = _numeric_dict(res.get("val_metrics", {}))
            test = _numeric_dict(res.get("test_metrics", {}))
            extra = {}
        else:  # regression
            val  = _numeric_dict(res.get("val_metrics_reg", {}))
            test = _numeric_dict(res.get("test_metrics_reg", {}))
            extra = {}
            if "regression_to_class" in res:
                extra["regression_to_class"] = res["regression_to_class"]  # 原樣放入，當報表

        folds_out.append({"fold_id": i, "val": val, "test": test, **extra})

    # 置頂 test 平均 / 標準差（自動對所有數值欄）
    test_avg, test_std = _avg_std_dict([f["test"] for f in folds_out])
    val_avg,  val_std  = _avg_std_dict([f["val"]  for f in folds_out])

    summary = {
        "task_type": task_type,
        "test_avg": test_avg,         # ★ 置頂
        "test_std": test_std,
        "val_avg": val_avg,           #（可選）一起放，方便比對
        "val_std": val_std,
        "folds": folds_out
    }
    with open(Path(export_dir) / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

















# =========================
# Optuna 最佳設定輸出（保持）
# =========================
def dump_best_yaml(study: optuna.Study, cfg: dict, run_dir: Path):
    """將最佳 trial 的參數與設定儲存成 YAML 檔與 txt 檔。"""
    best = study.best_trial
    params = best.params
    feats = best.user_attrs.get("selected_features", [])

    outdir = _ensure_dir(Path(run_dir) / "best")

    # 1) 純參數
    with open(outdir / "best_params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)

    # 2) 完整 config（將搜尋空間替換為實際值）
    frozen = copy.deepcopy(cfg)

    def _set_num(node, key, default):
        val = params.get(key, None)
        if val is None:
            return default
        try:
            v = float(val)
            if v.is_integer(): v = int(v)
        except Exception:
            v = val
        node[key] = v
        return v

    _set_num(frozen["train"], "lr", frozen["train"]["lr"])
    _set_num(frozen["train"], "weight_decay", frozen["train"]["weight_decay"])
    _set_num(frozen["train"], "epochs", frozen["train"]["epochs"])
    _set_num(frozen["train"], "grad_clip", frozen["train"]["grad_clip"])

    if "hidden_size" in params: frozen["model"]["hidden_size"] = params["hidden_size"]
    if "n_layers" in params:    frozen["model"]["n_layers"]    = params["n_layers"]
    if "dropout" in params:     frozen["model"]["dropout"]     = float(params["dropout"])
    if "seq_len" in params:     frozen["sequence"]["seq_len"]  = params["seq_len"]
    if "flat_band_bps" in params: frozen["label"]["flat_band_bps"] = params["flat_band_bps"]

    sel = frozen.setdefault("features", {}).setdefault("selection", {})
    if "k_features" in params: sel["k_range"] = [params["k_features"], params["k_features"]]
    if "feat_seed" in params:  sel["feat_seed"] = params["feat_seed"]

    with open(outdir / "selected_features.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feats))

    with open(outdir / "best_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(frozen, f, sort_keys=False, allow_unicode=True)

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
