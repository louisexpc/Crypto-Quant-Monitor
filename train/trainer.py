import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch import amp
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from export_metrices import save_fold_metrics

def multiclass_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "acc": acc,
        "macro_precision": p_macro, "macro_recall": r_macro, "macro_f1": f1_macro,
        "weighted_precision": p_weight, "weighted_recall": r_weight, "weighted_f1": f1_weight,
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
    
    
    assert torch.cuda.is_available(), "此版本統一使用 CUDA AMP，請在有 GPU 的環境執行。"
    device = device or "cuda"
    torch.set_float32_matmul_precision("high")  # 允許 TF32

    lr        = float(cfg["train"]["lr"])
    wd        = float(cfg["train"]["weight_decay"])
    clip      = float(cfg["train"]["grad_clip"])
    epochs    = int(cfg["train"]["epochs"])
    patience  = int(cfg["train"]["early_stopping_patience"])
    num_class = int(cfg["model"].get("num_classes", 3))

    model = model.to(device)

    # AdamW with fused if available
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd,
        fused=True if torch.cuda.is_available() else False
    )
    scaler = amp.GradScaler(device="cuda", enabled=True)
    dtype = _amp_dtype()

    # 類別權重（訓練前估一次）
    class_weights = _infer_class_weights(train_loader, num_class, device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    best_val_prec = -1.0
    best_epoch  = 0
    best_state  = copy.deepcopy(model.state_dict())
    wait = 0

    history = []
    printed_shape = False
    prefix = f"[fold {fold_id}] " if fold_id is not None else ""

    for epoch in range(1, epochs + 1):
        # -------------------- TRAIN --------------------
        model.train()
        train_loss_sum, train_n = 0.0, 0
        tr_preds, tr_tgts = [], []

        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)        # [B, T, F]
            yb = yb.to(device, non_blocking=True).long() # [B]

            if not printed_shape:
                print("DEBUG batch X shape =", tuple(Xb.shape))
                printed_shape = True

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
                logits = model(Xb)               # [B, C]
                loss   = ce_loss(logits, yb)

            scaler.scale(loss).backward()
            if clip and clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()

            bs = Xb.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

            preds = logits.argmax(dim=-1)
            tr_preds.append(preds.detach().cpu())
            tr_tgts.append(yb.detach().cpu())

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
        va_preds, va_tgts = [], []
        with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()
                logits = model(Xb)
                loss   = ce_loss(logits, yb)
                bs = Xb.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs
                preds = logits.argmax(dim=-1)
                va_preds.append(preds.detach().cpu())
                va_tgts.append(yb.detach().cpu())

        if val_n > 0 and va_preds:
            avg_va_loss = val_loss_sum / val_n
            y_va = torch.cat(va_tgts).numpy()
            yhat_va = torch.cat(va_preds).numpy()
            m_va = multiclass_metrics(y_va, yhat_va)
        else:
            avg_va_loss = float("nan")
            m_va = { "acc": float("nan"), "macro_f1": -1.0,
                     "macro_precision": float("nan"), "macro_recall": float("nan"),
                     "weighted_f1": float("nan") }

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
        })

        print(f"{prefix}[Epoch {epoch:03d}] tr_loss={avg_tr_loss:.4f}"
              f"val_loss={avg_va_loss:.4f} | val_f1={m_va['macro_f1']:.3f} | val_acc={m_va['acc']:.3f} | val_precision={m_va['macro_precision']:.3f}")

        improved = m_va["macro_precision"] > (best_val_prec + 1e-6)
        if epoch == 1 or improved:
            best_val_f1 = m_va["macro_f1"]
            best_val_prec = m_va["macro_precision"]
            best_epoch  = epoch
            best_state  = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"{prefix}Early stop at epoch {epoch} | best_epoch={best_epoch} | val_macro_f1={best_val_f1:.4f} | val_macro_prec={best_val_prec:.4f}")
                break

    # 載回最佳權重
    model.load_state_dict(best_state)

    # -------------------- TEST --------------------
    model.eval()
    te_preds, te_tgts, test_loss_sum, test_n = [], [], 0.0, 0
    with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()
            logits = model(Xb)
            loss = ce_loss(logits, yb)
            bs = Xb.size(0)
            test_loss_sum += loss.item() * bs
            test_n += bs
            te_preds.append(logits.detach().argmax(dim=1).cpu())
            te_tgts.append(yb.detach().cpu())

    if test_n > 0 and te_preds:
        avg_te_loss = test_loss_sum / test_n
        y_te = torch.cat(te_tgts).numpy()
        yhat_te = torch.cat(te_preds).numpy()
        m_te = multiclass_metrics(y_te, yhat_te)
    else:
        avg_te_loss = float("nan")
        m_te = { "acc": float("nan"), "macro_f1": float("nan"),
                 "macro_precision": float("nan"), "macro_recall": float("nan"),
                 "weighted_f1": float("nan") }

    print(f"{prefix}TEST  acc={m_te['acc']:.3f} macro_f1={m_te['macro_f1']:.3f} "
          f"macro_p={m_te['macro_precision']:.3f} macro_r={m_te['macro_recall']:.3f}\n")

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

    save_fold_metrics(history, save_dir=export_dir, prefix=f"fold_{fold_id}_")
    torch.save(result, export_dir / f"fold_{fold_id}_result.pt")
    return model, result
