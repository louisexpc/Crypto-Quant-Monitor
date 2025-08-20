import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

from .trainer_base import (
    get_task_type, amp_dtype, build_optimizer, build_warmup_scheduler, build_grad_scaler,
    infer_class_weights, fit_temperature_ce, find_best_threshold_by_auc
)

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 匯出/視覺化/度量
from utils.compute_export_metrices import (
    save_fold_metrics, save_result, compute_metrics, plot_test_eval
)

# -----------------------------
# 分類專屬 Loss（CE / Focal）
# -----------------------------
class FocalLossCE(nn.Module):
    """Softmax 版 Focal Loss（支援 class weight）"""
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [B, C], target: [B] long
        logp = F.log_softmax(logits, dim=1)           # [B, C]
        p    = logp.exp()
        loss = -((1 - p) ** self.gamma) * logp        # [B, C]
        loss = loss.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean() if self.reduction == "mean" else loss.sum()

def build_classification_loss(cfg, class_weights):
    """分別支援 CE / Focal-CE，CE 可搭 label_smoothing。"""
    ls = float(cfg["train"].get("label_smoothing", 0.0))
    use_focal = bool(cfg["train"].get("use_focal", False))
    if use_focal:
        return FocalLossCE(gamma=float(cfg["train"].get("focal_gamma", 2.0)),
                           weight=class_weights, reduction="mean")
    else:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)

# -----------------------------
# 主訓練流程（分類）
# -----------------------------
def train_one_fold(
    model,
    train_loader,
    val_loader,
    test_loader,
    cfg,
    device: str = None,
    fold_id: int | None = None,
    export_dir: str | None = None
):
    if len(train_loader) == 0:
        print("[ERROR][trainer_cls] empty train loader")
        return None, None

    assert torch.cuda.is_available(), "[trainer_cls] 需要 CUDA GPU 環境。"
    device = device or "cuda"

    # 讀設定
    lr        = float(cfg["train"]["lr"])
    clip      = float(cfg["train"]["grad_clip"])
    epochs    = int(cfg["train"]["epochs"])
    patience  = int(cfg["train"]["early_stopping_patience"])
    num_class = int(cfg["model"]["num_classes"])
    primary   = (cfg.get("task", {}) or {}).get("primary_metric", "val_loss").lower()

    model = model.to(device)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_warmup_scheduler(optimizer, steps_per_epoch, cfg)
    dtype = amp_dtype()
    scaler = build_grad_scaler(dtype)

    # 建 loss 與權重
    # 先確認輸出維度
    model.eval()
    with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
        xb0, _ = next(iter(train_loader))
        xb0 = xb0[:1].to(device, non_blocking=True)
        logits0 = model(xb0)
        out_dim = int(logits0.shape[-1])
    model.train()
    if out_dim != num_class:
        raise ValueError(
            f"[trainer_cls] 模型 out_dim={out_dim} != num_classes={num_class}。請調整 head。"
        )
    class_weights = infer_class_weights(train_loader, num_class, device)
    loss_fn = build_classification_loss(cfg, class_weights)

    # Early-stop 狀態
    best_epoch = 0
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    wait = 0
    prefix = f"[fold {fold_id}] " if fold_id is not None else ""
    history = []

    best_cls_val_loss = float("inf")
    best_val_f1 = -1.0
    best_val_f_05 = -1.0
    best_val_prec = -1.0
    best_val_recall = -1.0
    best_val_thresh = None
    best_T = torch.tensor(1.0, device=device)

    printed_shape = False

    # ----------------- Epoch Loop -----------------
    for epoch in range(1, epochs + 1):
        # -------- TRAIN --------
        model.train()
        train_loss_sum, train_n = 0.0, 0
        tr_preds, tr_tgts = [], []

        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()

            if not printed_shape:
                print("[trainer_cls] DEBUG batch X shape =", tuple(Xb.shape))
                printed_shape = True

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
                logits = model(Xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            if clip and clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            bs = Xb.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

            preds = logits.argmax(dim=-1)
            tr_preds.append(preds.detach().cpu())
            tr_tgts.append(yb.detach().cpu())

        avg_tr_loss = train_loss_sum / max(1, train_n)
        y_tr = torch.cat(tr_tgts).numpy()
        yhat_tr = torch.cat(tr_preds).numpy()
        m_tr = compute_metrics(y_tr, yhat_tr)

        # -------- VAL --------
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        val_tgts = []
        val_logits = []

        with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()
                logits = model(Xb)
                loss = loss_fn(logits, yb)

                bs = Xb.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs
                val_tgts.append(yb.detach().cpu())
                val_logits.append(logits.detach().cpu())

        avg_va_loss = val_loss_sum / max(1, val_n)
        y_va = torch.cat(val_tgts, dim=0).numpy()
        logits_all = torch.cat(val_logits, dim=0).to(device)  # [N, C]

        # 溫度校準（該 epoch 的 T）
        if cfg["train"].get("use_temperature", True):
            T_epoch = fit_temperature_ce(logits_all, torch.from_numpy(y_va).to(device))
        else:
            T_epoch = torch.tensor(1.0, device=device)

        probs_all = F.softmax(logits_all / T_epoch, dim=1).cpu().numpy()

        # Binary：找最佳 threshold；multi-class 直接 argmax
        if probs_all.shape[1] == 2:
            y_score_va = probs_all[:, 1]
            cfg_thresh = cfg["train"].get("threshold", None)
            if cfg_thresh is None:
                curr_val_thresh, _ = find_best_threshold_by_auc(y_va, y_score_va)
            else:
                curr_val_thresh = float(cfg_thresh)
            yhat_va = (y_score_va >= curr_val_thresh).astype(int)
        else:
            yhat_va = probs_all.argmax(axis=1)
            curr_val_thresh = None

        m_va = compute_metrics(y_va, yhat_va)

        # Early Stopping：預設看 F0.5 或 val_loss
        primary = (cfg.get("task", {}) or {}).get("primary_metric", "val_loss").lower()
        if primary in ["f05_macro", "f_05_macro"]:
            improved = (m_va.get("f_05_macro", -1.0) > (best_val_f_05 + 1e-6))
        else:
            improved = (avg_va_loss < (best_cls_val_loss - 1e-6))

        if epoch == 1 or improved:
            best_val_f1 = m_va.get("macro_f1", best_val_f1)
            best_val_f_05 = m_va.get("f_05_macro", best_val_f_05)
            best_val_prec = m_va.get("macro_precision", best_val_prec)
            best_val_recall = m_va.get("macro_recall", best_val_recall)
            best_val_thresh = float(curr_val_thresh) if curr_val_thresh is not None else None
            best_cls_val_loss = avg_va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_T = T_epoch.detach().clone()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"{prefix}Early stop at epoch {epoch} | best_epoch={best_epoch} | val_f1={best_val_f1:.4f} | val_f05={best_val_f_05:.4f}")
                break

        history.append({
            "epoch": epoch,
            "train_loss": avg_tr_loss, "val_loss": avg_va_loss,
            "train_acc": m_tr.get("acc", np.nan),  "val_acc":  m_va.get("acc", np.nan),
            "train_macro_f1": m_tr.get("macro_f1", np.nan), "val_macro_f1": m_va.get("macro_f1", np.nan),
            "train_macro_precision": m_tr.get("macro_precision", np.nan),
            "val_macro_precision":   m_va.get("macro_precision", np.nan),
            "train_macro_recall":    m_tr.get("macro_recall", np.nan),
            "val_macro_recall":      m_va.get("macro_recall", np.nan),
            "val_f_05_macro":        m_va.get("f_05_macro", np.nan),
        })
        print(f"{prefix}[Epoch {epoch:03d}] tr_loss={avg_tr_loss:.4f} | val_loss={avg_va_loss:.4f} | val_acc={m_va.get('acc',np.nan):.3f} | val_f1={m_va.get('macro_f1',np.nan):.3f} | val_f05={m_va.get('f_05_macro',np.nan):.3f}")

    # 載回最佳權重
    model.load_state_dict(best_state)
    model.eval()

    # -------- TEST --------
    te_tgts, te_probs = [], []
    test_loss_sum, test_n = 0.0, 0

    loss_fn = build_classification_loss(cfg, infer_class_weights(train_loader, num_class, device))
    with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()
            logits = model(Xb)
            loss = loss_fn(logits, yb)

            bs = Xb.size(0)
            test_loss_sum += loss.item() * bs
            test_n += bs

            probs = F.softmax(logits / best_T, dim=1)
            te_probs.append(probs.detach().cpu())
            te_tgts.append(yb.detach().cpu())

    avg_te_loss = test_loss_sum / max(1, test_n)

    y_te = torch.cat(te_tgts).numpy()
    y_prob_te = torch.cat(te_probs).numpy()
    yhat_te = y_prob_te.argmax(axis=1)
    m_te = compute_metrics(y_te, yhat_te)
    print(f"{prefix}Test_acc={m_te['acc']:.3f} | test_f1={m_te['macro_f1']:.3f} | test_f05={m_te['f_05_macro']:.3f} | test_prec={m_te['macro_precision']:.3f} | test_re={m_te['macro_recall']:.3f} | mcc={m_te['mcc']:.3f}")

    # 二分類的 threshold 報表
    best_thresh = None
    if y_prob_te.shape[1] == 2:
        y_score = y_prob_te[:, 1]
        cfg_thresh = cfg["train"].get("threshold", None)
        if cfg_thresh is not None:
            best_thresh = float(cfg_thresh)
        elif best_val_thresh is not None:
            best_thresh = float(best_val_thresh)
        elif cfg["train"].get("allow_test_threshold_fallback", False):
            best_thresh, _ = find_best_threshold_by_auc(y_te, y_score)
        else:
            best_thresh = 0.5

    result = {
        "history": history,
        "best_epoch": best_epoch,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "test_metrics": {
            "test_loss": avg_te_loss,
            "test_acc": m_te["acc"],
            "test_macro_f1": m_te["macro_f1"],
            "test_macro_precision": m_te["macro_precision"],
            "test_macro_recall": m_te["macro_recall"],
            "test_weighted_f1": m_te["weighted_f1"],
            "test_MCC": m_te["mcc"]
        },
        "best_val_thresh": float(best_val_thresh) if best_val_thresh is not None else None,
        "temperature": float(best_T.item()) if torch.is_tensor(best_T) else 1.0
    }
    if y_prob_te.shape[1] == 2:
        yhat_thresh = (y_prob_te[:, 1] >= best_thresh).astype(int)
        m_thresh = compute_metrics(y_te, yhat_thresh)
        result["threshold_metrics"] = {
            "best_threshold": float(best_thresh),
            "acc": m_thresh["acc"],
            "macro_f1": m_thresh["macro_f1"],
            "macro_precision": m_thresh["macro_precision"],
            "macro_recall": m_thresh["macro_recall"],
            "f_05_macro": m_thresh["f_05_macro"],
        }

    # 匯出與作圖
    save_fold_metrics(history, save_dir=export_dir, prefix=f"fold_{fold_id}_")
    save_result(fold_id=fold_id, export_dir=export_dir, result=result)
    plot_test_eval(
        y_true=y_te, y_pred=yhat_te, y_prob=y_prob_te,
        class_names=cfg["model"].get("class_names", None),
        save_dir=export_dir,
        prefix=f"fold_{fold_id}_",
        threshold=float(best_thresh) if (y_prob_te.shape[1] == 2 and best_thresh is not None) else None
    )
    return model, result
