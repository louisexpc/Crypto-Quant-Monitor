# train/evaluation/reporters/classification_reporter.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, \
    confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from ._utils import _ensure_dir, _ensure_2d_prob

class ClassificationReporter:
    """
    1. 說明: 分類任務可視化與摘要（ROC/PR/ConfMat + AUC/AP JSON）
    2. inputs: 於 __init__ 與 plot_eval 傳入
    3. return: plot_eval 回傳 dict 摘要（包含各類別/micro AUC、AP 與門檻點資訊）
    """
    def __init__(self, save_dir, prefix: str = "", class_names: Optional[List[str]] = None):
        """
        1. 說明: 初始化 Reporter（設定輸出路徑、檔名前綴與類別名稱）
        2. inputs:
           - save_dir: 圖表/JSON 的輸出資料夾
           - prefix: 檔名前綴（例如 "test_"）
           - class_names: 類別名稱清單（可為 None）
        3. return: None
        """
        self.save_dir = _ensure_dir(save_dir)
        self.prefix = prefix or ""
        self.class_names = list(class_names) if class_names is not None else None

    def plot_eval(self, y_true, y_pred, y_prob, threshold: float | None = None) -> Dict[str, Any]:
        """
        1. 說明: 輸出 ROC/PR/混淆矩陣 圖與 AUC/AP 摘要 JSON
        2. inputs:
           - y_true: [N] 真實標籤
           - y_pred: [N] 預測類別
           - y_prob: [N] 或 [N,1] 或 [N,C] 預測機率
           - threshold: 二分類時在 PR 曲線上標記的門檻（可選）
        3. return:
           - dict: {class_name: {roc_auc, pr_auc}, "micro_avg": {...}, "threshold": {...}}
        """
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
            Y = np.concatenate([1 - Y, Y], axis=1)
        if n_classes == 2 and y_prob.shape[1] == 1:
            p1 = y_prob[:, 0]
            y_prob = np.stack([1.0 - p1, p1], axis=1)

        # ROC
        roc_auc: Dict[str, float] = {}
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

        # PR
        pr_auc: Dict[str, float] = {}
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
                idx = np.argmin(np.abs(thr - threshold)) if len(thr) else None
                if idx is not None and 0 <= idx < len(prec):
                    p_thr, r_thr = prec[idx], rec[idx]
                    plt.plot(r_thr, p_thr, "o", markersize=8)
                    plt.annotate(
                        f"thr={threshold}\nP={p_thr:.2f}, R={r_thr:.2f}",
                        (r_thr, p_thr), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9, arrowprops=dict(arrowstyle="->")
                    )
                    thr_point = {"precision": float(p_thr), "recall": float(r_thr)}

        prec_m, rec_m, _ = precision_recall_curve(Y.ravel(), y_prob.ravel())
        ap_m = average_precision_score(Y.ravel(), y_prob.ravel())
        pr_auc["micro_avg"] = float(ap_m)
        plt.plot(rec_m, prec_m, "--", label=f"micro-avg (AP={ap_m:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall (OvR)")
        plt.legend(); plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}pr_curve.png", dpi=200)
        plt.close()

        # Confusion Matrix（row-normalized）
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)), normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap="Blues", xticks_rotation=45, values_format=".2f")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.prefix}confusion_matrix.png", dpi=200)
        plt.close()

        # JSON 摘要
        summary: Dict[str, Any] = {}
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
