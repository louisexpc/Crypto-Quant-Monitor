# config_export.py

import yaml
import copy
from pathlib import Path
import optuna
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score
)
from sklearn.preprocessing import label_binarize

def dump_best_yaml(study: optuna.Study, cfg: dict, run_dir: Path):
    """將最佳 trial 的參數與設定儲存成 YAML 檔與 txt 檔。"""
    best = study.best_trial
    params = best.params
    feats = best.user_attrs.get("selected_features", [])

    outdir = run_dir / "best"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. 純參數 YAML
    with open(outdir / "best_params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)

    # 2. 完整 config（將搜尋空間替換為實際值）
    frozen = copy.deepcopy(cfg)

    def _set_num(node, key, default):
        val = params.get(key, None)
        if val is None:
            return default
        try:
            v = float(val)
            if v.is_integer():
                v = int(v)
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

    # 3. 寫出 selected_features.txt
    with open(outdir / "selected_features.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feats))

    # 4. 寫出 best_config.yaml
    with open(outdir / "best_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(frozen, f, sort_keys=False, allow_unicode=True)

def save_fold_metrics(metrics: list[dict], save_dir: Path, prefix: str = "", y_true=None, y_pred=None, class_names=None):
    """
    儲存單個 fold 的訓練過程：
    - metrics: list of dicts，每個 epoch 的訓練記錄
    - save_dir: 儲存路徑
    - prefix: 檔名前綴（如 fold_0）

    會儲存：
    - metrics_epoch.csv
    - loss_curve.png
    - accuracy_curve.png
    - precision_recall_curve.png
    """

    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv(save_dir / f"{prefix}metrics_epoch.csv", index=False)

    # ----- Loss curve -----
    plt.figure()
    plt.plot(df["train_loss"], label="Train Loss")
    plt.plot(df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}loss_curve.png")
    plt.close()

    # ----- Accuracy curve -----
    plt.figure()
    plt.plot(df["train_acc"], label="Train Acc")
    plt.plot(df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}accuracy_curve.png")
    plt.close()

    # # ----- Precision / Recall curve -----
    # plt.figure()
    # # plt.plot(df["train_macro_precision"], label="Train Precision")
    # plt.plot(df["val_macro_precision"], label="Val Precision")
    # # plt.plot(df["train_macro_recall"], label="Train Recall")
    # plt.plot(df["val_macro_recall"], label="Val Recall")
    # plt.xlabel("Epoch")
    # plt.ylabel("Score")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(save_dir / f"{prefix}precision_recall_curve.png")
    # plt.close()

    # ----- f1 curve -----
    plt.figure()
    plt.plot(df["train_macro_f1"], label="Train f1")
    plt.plot(df["val_macro_f1"], label="Val f1")
    plt.xlabel("Epoch")
    plt.ylabel("f1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}f1_curve.png")
    plt.close()

    if y_true is not None and y_pred is not None:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}confusion_matrix.png")
        plt.close()


def _ensure_2d_prob(y_prob):
    """
    將 y_prob 正規化為 2D:
    - (N,)  -> (N,1)
    - (N,1) 保持
    - (N,2+) 保持
    """
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, 1)
    return y_prob


def _ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

import json
def plot_test_eval(
    y_true,
    y_pred,
    y_prob,
    save_dir,
    prefix="",
    class_names=None,
    threshold: float | None = None  # ✅ 新增參數
):
    """
    繪製：
    - ROC 曲線
    - PR 曲線
    - Confusion Matrix
    並儲存為圖與 CSV。
    """
    save_dir = _ensure_dir(save_dir)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = _ensure_2d_prob(y_prob)

    # 固定 classes，避免因缺類別導致欄數縮水
    n_classes = len(class_names)
    Y = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2 and Y.shape[1] == 1:
        p1 = y_prob[:, 0]
        y_prob = np.stack([1.0 - p1, p1], axis=1)

    # 檢查 y_true 是否都在 [0..n_classes-1]
    uniq = np.unique(y_true)
    assert uniq.min() >= 0 and uniq.max() < n_classes, \
        f"y_true 的類別值需落在 [0..{n_classes-1}]，但目前 unique={uniq}。請先在前處理 map 成連續整數。"

    # One-vs-Rest binarize
    Y = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2 and Y.shape[1] == 1:
        Y = np.concatenate([1 - Y, Y], axis=1)
   

    # ROC
    roc_auc = {}
    plt.figure(figsize=(6,5))
    for c in range(n_classes):
        if Y[:,c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(Y[:,c], y_prob[:,c])
        auc_c = auc(fpr, tpr)
        roc_auc[class_names[c]] = float(auc_c)
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc_c:.3f})")

    fpr_m, tpr_m, _ = roc_curve(Y.ravel(), y_prob.ravel())
    auc_m = auc(fpr_m, tpr_m)
    roc_auc["micro_avg"] = float(auc_m)
    plt.plot(fpr_m, tpr_m, "--", label=f"micro-avg (AUC={auc_m:.3f})")
    plt.plot([0,1],[0,1], ":", label="chance")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC (OvR)"); plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}roc_curve.png", dpi=200)
    plt.close()

    # --------------------
    # PR (OvR + micro)
    # --------------------
    pr_auc = {}
    plt.figure(figsize=(6, 5))
    for c in range(n_classes):
        if Y[:, c].sum() == 0:
            continue
        prec, rec, thr = precision_recall_curve(Y[:, c], y_prob[:, c])
        ap = average_precision_score(Y[:, c], y_prob[:, c])
        pr_auc[class_names[c]] = float(ap)
        plt.plot(rec, prec, label=f"{class_names[c]} (AP={ap:.3f})")

        # 若是二分類且有給定 threshold，就標記出來
        if n_classes == 2 and threshold is not None and class_names[c] == class_names[-1]:
            # 在 PR curve 上找出接近指定 threshold 的位置
            idx = np.argmin(np.abs(thr - threshold))
            p_thr, r_thr = prec[idx], rec[idx]
            plt.plot(r_thr, p_thr, "o", markersize=8)
            plt.annotate(f"thr={threshold}\nP={p_thr:.2f}, R={r_thr:.2f}",
                         (r_thr, p_thr), xytext=(0,10), textcoords="offset points",
                         ha='center', fontsize=9, color='darkblue',
                         arrowprops=dict(arrowstyle="->"))
            thr_point = {"precision": float(p_thr), "recall": float(r_thr)}
        else:
            thr_point = None  # 若不是二分類就忽略

    # micro-average
    prec_m, rec_m, _ = precision_recall_curve(Y.ravel(), y_prob.ravel())
    ap_m = average_precision_score(Y.ravel(), y_prob.ravel())
    pr_auc["micro_avg"] = float(ap_m)
    plt.plot(rec_m, prec_m, "--", label=f"micro-avg (AP={ap_m:.3f})")

    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall (OvR)")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}pr_curve.png", dpi=200)
    plt.close()

    # --------------------
    # Confusion Matrix
    # --------------------
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}confusion_matrix.png")
    plt.close()

    # --------------------
    # JSON summary（AUC + AP + micro + threshold）
    # --------------------
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

    with open(save_dir / f"{prefix}auc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary