import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch import amp
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve, fbeta_score
from export_metrices import save_fold_metrics, plot_test_eval
import time

from utils.cuda_utils import setup_cuda_acceleration
setup_cuda_acceleration()



def find_best_threshold_by_auc(y_true, y_prob_pos):
    """
    對二分類的預測機率 y_prob_pos，找出最佳 threshold（根據 Youden 指標）

    y_true: 真實標籤（0 or 1）
    y_prob_pos: 機率（為 class=1 的 softmax 或 sigmoid 輸出）
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_threshold = thresholds[best_idx]
    
    return best_threshold, {
        "best_idx": best_idx,
        "fpr": fpr[best_idx],
        "tpr": tpr[best_idx],
        "threshold": best_threshold,
    }

def multiclass_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    f05_macro = fbeta_score(y_true, y_pred, beta=0.5, average="macro", zero_division=0)
    f05_weighted = fbeta_score(y_true, y_pred, beta=0.5, average="weighted", zero_division=0)
    return {
        "acc": acc,
        "macro_precision": p_macro, "macro_recall": r_macro, "macro_f1": f1_macro,
        "weighted_precision": p_weight, "weighted_recall": r_weight, "weighted_f1": f1_weight,
        "f_05_macro": f05_macro, "f_05_weighted": f05_weighted,
    }

@torch.no_grad()
def _infer_class_weights(train_loader, num_classes: int, device: str):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    any_batch = False
    for _, yb in train_loader:
        any_batch = True
        binc = torch.bincount(yb.view(-1).cpu(), minlength=num_classes).to(torch.float64)
        counts[:num_classes] += binc[:num_classes]
    if not any_batch:
        # 防訓練集為空
        return torch.ones(num_classes, device=device, dtype=torch.float32)
    counts = counts.clamp(min=1)  # 防 0
    total = counts.sum()
    weights = (total / counts).to(torch.float32)
    return weights.to(device)

def _amp_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def train_one_fold(
    model,
    train_loader,
    val_loader,
    test_loader,
    cfg,
    device: str = None,
    fold_id: int | None = None,
    export_dir: str = None
):
    if len(train_loader) == 0:
        print("[ERROR] empty train loader")
        return None, None  # 或 raise Exception("...")
    
    assert torch.cuda.is_available(), "此版本統一使用 CUDA AMP，請在有 GPU 的環境執行。"
    device = device or "cuda"
    


    lr        = float(cfg["train"]["lr"])
    wd        = float(cfg["train"]["weight_decay"])
    clip      = float(cfg["train"]["grad_clip"])
    epochs    = int(cfg["train"]["epochs"])
    patience  = int(cfg["train"]["early_stopping_patience"])
    num_class = int(cfg["model"]["num_classes"])

    model = model.to(device)

    # AdamW with fused if available
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd,
        fused=True if torch.cuda.is_available() else False
    )

    # -------- Warmup Scheduler --------
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs

    # 支援 warmup_steps（優先）或 warmup_ratio
    if "warmup_steps" in cfg["train"]:
        warmup_steps = int(cfg["train"]["warmup_steps"]) * steps_per_epoch
    else:
        warmup_ratio = float(cfg["train"].get("warmup_ratio", 0.1))
        warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            return 1.0  # 可選 linear decay: (total_steps - step) / (total_steps - warmup_steps)

    scheduler = LambdaLR(optimizer, lr_lambda)

    scaler = amp.GradScaler(enabled=True)
    dtype = _amp_dtype()

    # 類別權重（訓練前估一次）
    class_weights = _infer_class_weights(train_loader, num_class, device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    # 用於early stopping + 紀錄最佳指標
    best_val_f1 = -1.0
    best_val_prec = -1.0
    best_val_recall = -1.0
    best_val_f_05 = -0.1
    best_val_loss = 500000000
    best_epoch  = 0
    best_state  = copy.deepcopy(model.state_dict())
    wait = 0

    history = []
    printed_shape = False
    prefix = f"[fold {fold_id}] " if fold_id is not None else ""

    for epoch in range(1, epochs + 1):
        # -------------------- TRAIN --------------------
        model.train()
        t0 = time.time()
        data_wait = 0.0
        compute_t = 0.0
        it_time = time.time()

        train_loss_sum, train_n = 0.0, 0
        tr_preds, tr_tgts = [], []

        for it, (Xb, yb) in enumerate(train_loader):
            data_wait += time.time() - it_time

            Xb = Xb.to(device, non_blocking=True)        # [B, T, F]
            yb = yb.to(device, non_blocking=True).long() # [B]

            if not printed_shape:
                print("DEBUG batch X shape =", tuple(Xb.shape))
                printed_shape = True

            optimizer.zero_grad(set_to_none=True)

            ct0 = time.time()   

            with amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
                logits = model(Xb)               # [B, C]
                loss   = ce_loss(logits, yb)
            scaler.scale(loss).backward()

            if clip and clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            compute_t += time.time() - ct0     # ← 累計前向+反向時間

            bs = Xb.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

            preds = logits.argmax(dim=-1)
            tr_preds.append(preds.detach().cpu())
            tr_tgts.append(yb.detach().cpu())

            it_time = time.time()
        # print(f"[Epoch {epoch:03d}] ⏱ Data Wait: {data_wait:.2f}s | Compute: {compute_t:.2f}s | Total: {time.time()-t0:.2f}s")

        # 訓練集度量（防空）
        if train_n > 0 and tr_preds:
            avg_tr_loss = train_loss_sum / train_n
            y_tr = torch.cat(tr_tgts).numpy()
            yhat_tr = torch.cat(tr_preds).numpy()
            m_tr = multiclass_metrics(y_tr, yhat_tr)
        else:
            avg_tr_loss = float("nan")
            m_tr = { "acc": float("nan"), "macro_f1": float("nan"),
                     "macro_precision": float("nan"), "macro_recall": float("nan"),
                     "weighted_f1": float("nan") }

        # -------------------- VAL --------------------
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        val_probs, val_tgts = [], []
        with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()
                logits = model(Xb)
                loss   = ce_loss(logits, yb)
                
                bs = Xb.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs
                val_tgts.append(yb.detach())


                if logits.size(1) == 1:
                    p1 = torch.sigmoid(logits).squeeze(1)
                    probs = torch.stack([1.0-p1, p1], dim = 1)
                else:
                    probs = F.softmax(logits, dim=1)
                val_probs.append(probs.detach())

            avg_va_loss = val_loss_sum / val_n
            y_va = torch.cat(val_tgts, dim=0).cpu().numpy()
            y_prob_va = torch.cat(val_probs, dim=0).cpu().numpy()
            y_score_va = y_prob_va[:, 1]

            val_thresh = cfg["train"].get("threshold", 0.5)
            yhat_va = (y_score_va >= val_thresh).astype(int)
            m_va = multiclass_metrics(y_va, yhat_va)


        history.append({
            "epoch": epoch,
            "train_loss": avg_tr_loss, "val_loss": avg_va_loss,
            "train_acc": m_tr["acc"],  "val_acc":  m_va["acc"],
            "train_macro_f1": m_tr["macro_f1"], "val_macro_f1": m_va["macro_f1"],
            "train_macro_precision": m_tr.get("macro_precision", float("nan")),
            "val_macro_precision":   m_va.get("macro_precision", float("nan")),
            "train_macro_recall":    m_tr.get("macro_recall", float("nan")),
            "val_macro_recall":      m_va.get("macro_recall", float("nan")),
            "train_weighted_f1":     m_tr.get("weighted_f1", float("nan")),
            "val_weighted_f1":       m_va.get("weighted_f1", float("nan")),
            "val_f_05_macro ":       m_va.get("f_05_macro", float("nan")),
            "val_f_05_weighted":     m_va.get("f_05_weighted", float("nan")),
        })

        print(f"{prefix}[Epoch {epoch:03d}] tr_loss={avg_tr_loss:.4f} | "
              f"val_loss={avg_va_loss:.4f} | val_acc={m_va['acc']:.3f} | val_f1={m_va['macro_f1']:.3f} | val_f05={m_va['f_05_macro']:.3f} |val_prec={m_va['macro_precision']:.3f} | val_re={m_va["macro_recall"]:.3f}")

        # improved = m_va["macro_f1"] > (best_val_f1 + 1e-6)
        # improved = m_va["f_05_macro"] > (best_val_f_05 + 1e-6)
        improved = avg_va_loss < (best_val_loss - 1e-6)
        
        if epoch == 1 or improved:
            best_val_f1 = m_va["macro_f1"]
            best_val_f_05 = m_va["f_05_macro"]
            best_val_prec = m_va["macro_precision"]
            best_val_recall = m_va["macro_recall"]
            best_val_loss = avg_va_loss
            best_epoch  = epoch
            best_state  = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"{prefix}Early stop at epoch {epoch} | best_epoch={best_epoch} | val_f1={best_val_f1:.4f} | val_f_05={best_val_f_05:.4f} | val_prec={best_val_prec:.4f} | best_recall={best_val_recall:.4f}")
                break




    # 載回最佳權重
    model.load_state_dict(best_state)

    # -------------------- TEST --------------------
    model.eval()
    te_tgts, te_prob_chunks = [], []
    test_loss_sum, test_n = 0.0, 0

    with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()
            logits = model(Xb)
            loss = ce_loss(logits, yb)
            bs = Xb.size(0)
            test_loss_sum += loss.item() * bs
            test_n += bs

            # 儲存機率
            if logits.size(1) == 1: # shape = [B, 1]
                p1 = torch.sigmoid(logits).squeeze(1)
                probs = torch.stack([1.0 - p1, p1], dim=1)

            else:                   # shape = [B, num_classes]
                probs = F.softmax(logits, dim=1)
            te_prob_chunks.append(probs.detach().cpu())

            te_tgts.append(yb.detach().cpu())

    # === 統一後處理 ===
    avg_te_loss = test_loss_sum / test_n
    y_te = torch.cat(te_tgts).numpy()
    y_prob_te = torch.cat(te_prob_chunks).numpy()
    yhat_te = y_prob_te.argmax(axis=1)  # ← 原始 argmax 預測

    m_te = multiclass_metrics(y_te, yhat_te)

    if num_class == 2:
        y_score = y_prob_te[:, 1]
        val_thresh = cfg["train"].get("threshold", None)

        if val_thresh is not None:
            best_thresh = val_thresh
            roc_info = None
        else:
            best_thresh, roc_info = find_best_threshold_by_auc(y_te, y_score)

        yhat_thresh = (y_score >= best_thresh).astype(int)
        m_thresh = multiclass_metrics(y_te, yhat_thresh)

  
    print(f"{prefix}Tset_acc={m_te['acc']:.3f} | test_f1={m_te['macro_f1']:.3f} | test_f05={m_te['f_05_macro']:.3f} | test_prec={m_te['macro_precision']:.3f} | test_re={m_te["macro_recall"]:.3f}\n")


    result = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": {
            "test_loss": avg_te_loss,
            "test_acc": m_te["acc"],
            "test_macro_f1": m_te["macro_f1"],
            "test_macro_precision": m_te["macro_precision"],
            "test_macro_recall": m_te["macro_recall"],
            "test_weighted_f1": m_te["weighted_f1"],
        },
        "state_dict": best_state,
    }
    if num_class == 2:
        result["threshold_metrics"] = {
            "best_threshold": float(best_thresh),
            "acc": m_thresh["acc"],
            "macro_f1": m_thresh["macro_f1"],
            "macro_precision": m_thresh["macro_precision"],
            "macro_recall": m_thresh["macro_recall"],
            "f_05_macro": m_thresh["f_05_macro"],  # ← 新增
        }
        result["roc_info"] = roc_info  # 可選：儲存 fpr, tpr, thresholds

    save_fold_metrics(history, save_dir=export_dir, prefix=f"fold_{fold_id}_")
    plot_test_eval(
        y_true=y_te,
        y_pred=yhat_te,
        y_prob=y_prob_te,
        class_names=cfg["model"]["class_names"],
        save_dir=export_dir,
        prefix=f"fold_{fold_id}_",
        threshold=float(best_thresh) if num_class == 2 else None  
    )
    # torch.save(result, export_dir / f"fold_{fold_id}_result.pt")
    return model, result
