# trainer.py
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch import amp
from sklearn.metrics import roc_curve

# 匯出/視覺化/客製目標工具（已整合在 config_export.py）
from compute_export_metrices import (
    save_fold_metrics, save_result, compute_metrics, plot_test_eval,
    plot_regression_eval, plot_regression_threshold_sweep,
    compute_mixed_objective_np
)

# 回歸度量與損失建構（外部模組，沿用你的實作）
from regression_utils import (
    compute_regression_metrics, build_regression_loss
)

from utils.init_train import setup_cuda_acceleration
setup_cuda_acceleration()


# =========================
# 1) 公用：任務/度量/工具
# =========================
def get_task_type(cfg):
    """優先讀 cfg.task.type；否則看 num_classes 推斷。"""
    if "task" in cfg and "type" in cfg["task"]:
        return str(cfg["task"]["type"]).lower()
    if "target" in cfg and "type" in cfg["target"]:
        return str(cfg["target"]["type"]).lower()
    nc = int(cfg["model"].get("num_classes", 1))
    return "classification" if nc >= 2 else "regression"


def _amp_dtype():
    """優先 bf16（若支援），否則 fp16。"""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def _infer_class_weights(train_loader, num_classes: int, device: str):
    """
    根據訓練資料分佈產生 CE 的 class weight。
    counts -> total/counts（反比權重）
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, yb in train_loader:
        yb = yb.view(-1).cpu()
        binc = torch.bincount(yb, minlength=num_classes).to(torch.float64)
        counts[:num_classes] += binc[:num_classes]
    counts = counts.clamp(min=1)
    total = counts.sum()
    class_weights = (total / counts).to(torch.float32).to(device)
    return class_weights


def find_best_threshold_by_auc(y_true, y_prob_pos):
    """
    二分類：用 Youden J 指標（tpr - fpr）找最佳 threshold
    y_true: 0/1
    y_prob_pos: 對正類的機率（或得分）
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_threshold = thresholds[best_idx]
    return float(best_threshold), {
        "best_idx": int(best_idx),
        "fpr": float(fpr[best_idx]),
        "tpr": float(tpr[best_idx]),
        "threshold": float(best_threshold),
    }


# =========================
# 2) 損失：分類（CE / Focal-CE）
# =========================
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
    """
    分類統一用 CE；可切換為 Focal-CE。
    - label_smoothing 僅 CE 有效
    """
    ls = float(cfg["train"].get("label_smoothing", 0.0))
    use_focal = bool(cfg["train"].get("use_focal", False))
    if use_focal:
        return FocalLossCE(gamma=float(cfg["train"].get("focal_gamma", 2.0)),
                           weight=class_weights, reduction="mean")
    else:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)


# ---- CE 版溫度校準（直接對 logits/T 做 CE）----
def fit_temperature_ce(logits, y_true, max_iter=50):
    """
    logits: [N, C] (未經 softmax 的原始值)
    y_true: [N] long
    回傳溫度 T（torch.Tensor[1]）
    """
    T = torch.nn.Parameter(torch.ones(1, device=logits.device))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    y_true = y_true.to(logits.device).long()

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, y_true, reduction="mean")
        loss.backward()
        return loss

    opt.step(closure)
    return T.detach()


# =========================
# 3) 主流程
# =========================
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
        print("[ERROR] empty train loader")
        return None, None

    assert torch.cuda.is_available(), "此版本統一使用 CUDA AMP，請在有 GPU 的環境執行。"
    device = device or "cuda"
    if get_task_type(cfg) == "regression":
        xb_dbg, yb_dbg = next(iter(train_loader))
        # ★ 檢查不同樣本的 y 是否幾乎相同（避免 y 全常數）
        y_std = float(yb_dbg.float().std().item())
        if y_std == 0.0:
            print("[ALERT][trainer] train batch y std=0. 可能是 dataloader 標籤 dtype 或對齊問題。")

    # ----- 讀取設定 -----
    lr        = float(cfg["train"]["lr"])
    wd        = float(cfg["train"]["weight_decay"])
    clip      = float(cfg["train"]["grad_clip"])
    epochs    = int(cfg["train"]["epochs"])
    patience  = int(cfg["train"]["early_stopping_patience"])
    num_class = int(cfg["model"]["num_classes"])
    task_type = get_task_type(cfg)   # "classification" or "regression"
    primary   = (cfg.get("task", {}) or {}).get("primary_metric", "val_loss").lower()

    model = model.to(device)

    # AdamW with fused if available
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd,
        fused=True if torch.cuda.is_available() else False
    )

    # -------- Warmup Scheduler --------
    steps_per_epoch = len(train_loader)
    warmup_steps = int(cfg["train"]["warmup_epochs"]) * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            # 可選：這裡可以換成 linear decay 或 cosine
            return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
    dtype = _amp_dtype()
    scaler = amp.GradScaler(enabled=(dtype == torch.float16))

    # ===== Loss 建立 =====
    if task_type == "classification":
        # 試跑一個 batch 取得輸出維度
        xb0, _ = next(iter(train_loader))
        xb0 = xb0[:1].to(device, non_blocking=True)
        model.eval()
        with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
            logits0 = model(xb0)
        model.train()
        out_dim = int(logits0.shape[-1])
        if out_dim != num_class:
            raise ValueError(
                f"[分類任務] 模型輸出維度 out_dim={out_dim} 必須等於 num_classes={num_class}。"
                " 請調整模型 head（包含二分類時需輸出 2-logit）。"
            )
        class_weights = _infer_class_weights(train_loader, num_class, device)
        loss_fn = build_classification_loss(cfg, class_weights)
        loss_kind = "ce"
    else:
        loss_fn = build_regression_loss(cfg)  # α·EMA-MSE + β·(1-Pearson) / mse / huber 由 cfg 控制
        loss_kind = "reg"

    # ====== Early Stopping 狀態 ======
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    # 統一 history 記錄
    history = []
    prefix = f"[fold {fold_id}] " if fold_id is not None else ""

    # 分類：最佳度量
    best_cls_val_loss = float("inf")
    best_val_f1 = -1.0
    best_val_f_05 = -1.0
    best_val_prec = -1.0
    best_val_recall = -1.0
    best_val_thresh = None
    best_T = torch.tensor(1.0, device=device)  # 儲存「最佳 epoch」的 T

    # 回歸：最佳度量
    best_reg_val_loss = float("inf")
    best_reg_pearson  = -1.0
    best_reg_mixed    = float("inf")
    best_va_arrays    = None  # (y_va, y_pred_va) at best epoch

    printed_shape = False

    for epoch in range(1, epochs + 1):
        # -------------------- TRAIN --------------------
        model.train()
        train_loss_sum, train_n = 0.0, 0
        tr_preds, tr_tgts = [], []

        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb_t = yb.to(device, non_blocking=True).long() if task_type == "classification" else yb.to(device, non_blocking=True).float()

            # for debug 印出 dim
            if not printed_shape:
                print("DEBUG batch X shape =", tuple(Xb.shape))
                printed_shape = True

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
                logits = model(Xb)
                if task_type == "classification":
                    loss = loss_fn(logits, yb_t)
                else:
                    loss = loss_fn(logits.squeeze(-1), yb_t)

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

            if task_type == "classification":
                preds = logits.argmax(dim=-1)
                tr_preds.append(preds.detach().cpu())
                tr_tgts.append(yb_t.detach().cpu())
            else:
                tr_preds.append(logits.detach().float().cpu().squeeze(-1))
                tr_tgts.append(yb_t.detach().cpu())

        # 訓練集度量（防空）
        avg_tr_loss = train_loss_sum / max(1, train_n)
        if task_type == "classification":
            y_tr = torch.cat(tr_tgts).numpy()
            yhat_tr = torch.cat(tr_preds).numpy()
            m_tr = compute_metrics(y_tr, yhat_tr)
        else:
            y_tr = torch.cat(tr_tgts).numpy()
            yhat_tr = torch.cat(tr_preds).numpy()
            m_tr = compute_regression_metrics(y_tr, yhat_tr)

        # -------------------- VAL --------------------
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        val_tgts = []
        val_logits = []   # 分類：做溫度；回歸：不用
        val_chunks = []   # 分類：存 probs；回歸：存連續 y_pred

        with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb_t = yb.to(device, non_blocking=True).long() if task_type == "classification" else yb.to(device, non_blocking=True).float()
                logits = model(Xb)

                if task_type == "classification":
                    loss = loss_fn(logits, yb_t)
                else:
                    loss = loss_fn(logits.squeeze(-1), yb_t)

                bs = Xb.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs
                val_tgts.append(yb_t.detach().cpu())

                if task_type == "classification":
                    val_logits.append(logits.detach().cpu())
                else:
                    val_chunks.append(logits.detach().float().cpu().squeeze(-1))

        avg_va_loss = val_loss_sum / max(1, val_n)

        if task_type == "classification":
            # ======= 分類驗證 =======
            y_va = torch.cat(val_tgts, dim=0).numpy()
            logits_all = torch.cat(val_logits, dim=0).to(device)  # [N, C]

            # 溫度校準（可關閉；本 epoch 的 T）
            if cfg["train"].get("use_temperature", True):
                T_epoch = fit_temperature_ce(logits_all, torch.from_numpy(y_va).to(device))
            else:
                T_epoch = torch.tensor(1.0, device=device)

            probs_all = F.softmax(logits_all / T_epoch, dim=1).cpu().numpy()
            y_score_va = probs_all[:, 1] if probs_all.shape[1] == 2 else None

            # Binary：找最佳 threshold
            if probs_all.shape[1] == 2:
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

            # Early Stopping（預設看 F0.5；或改 val_loss）
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
                best_T = T_epoch.detach().clone()  # 綁定最佳權重的 T
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"{prefix}Early stop at epoch {epoch} | best_epoch={best_epoch} | val_f1={best_val_f1:.4f} | val_f_05={best_val_f_05:.4f}")
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

        else:
            # ======= 回歸驗證 =======


            y_va = torch.cat(val_tgts, dim=0).to(torch.float32).cpu().numpy()
            y_pred_va = torch.cat(val_chunks, dim=0).to(torch.float32).cpu().numpy()
            m_va = compute_regression_metrics(y_va, y_pred_va)

            # 以驗證分數組合為單一目標（Optuna 主目標 / 也可作 early-stop）
            alpha = float(cfg["loss"].get("alpha", 0.7))
            beta  = float(cfg["loss"].get("beta", 0.3))
            decay = float(cfg["loss"].get("ema_decay", 0.9))

            val_mixed, comps = compute_mixed_objective_np(
                y_true=y_va, y_pred=y_pred_va, alpha=alpha, beta=beta, ema_decay=decay
            )
            # 把三個指標塞進 m_va 以便記錄
            m_va["pearson_global"] = comps["pearson"]   # 整段驗證集 Pearson
            m_va["ema_mse_global"] = comps["ema_mse"]   # 整段驗證集 EMA-MSE
            m_va["mixed"]          = val_mixed          # α·EMA-MSE + β·(1−Pearson)

            if epoch == 1:
                std_pred = float(np.std(y_pred_va))
                r_sanity = float(np.corrcoef(y_va, y_pred_va)[0,1]) if len(y_va) > 2 else 0.0
                print(f"[DEBUG] val std(y_pred)={std_pred:.6g}, corr={r_sanity:.4f}")

            # Early-Stopping 規則
            if primary == "mixed":
                improved = (epoch == 1) or (val_mixed < (best_reg_mixed - 1e-6))
            elif primary == "pearson":
                improved = (epoch == 1) or (m_va["pearson"] > (best_reg_pearson + 1e-6))
            else:
                improved = (epoch == 1) or (avg_va_loss < (best_reg_val_loss - 1e-6))

            if improved:
                best_reg_val_loss = avg_va_loss
                best_reg_pearson  = m_va["pearson"]
                best_reg_mixed    = val_mixed
                best_epoch        = epoch
                best_state        = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_va_arrays    = (y_va.copy(), y_pred_va.copy())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"{prefix}Early stop at epoch {epoch} | best_epoch={best_epoch} | val_mixed={best_reg_mixed:.6g} | val_pearson={best_reg_pearson:.4f}")
                    break

            history.append({
                "epoch": epoch,
                "train_loss": avg_tr_loss, "val_loss": avg_va_loss,
                "train_mse": m_tr.get("mse", np.nan), "val_mse": m_va.get("mse", np.nan),
                "train_mae": m_tr.get("mae", np.nan), "val_mae": m_va.get("mae", np.nan),
                "train_rmse": m_tr.get("rmse", np.nan), "val_rmse": m_va.get("rmse", np.nan),
                "train_pearson": m_tr.get("pearson", np.nan), "val_pearson": m_va.get("pearson", np.nan),
                "val_pearson_global": m_va.get("pearson_global", np.nan),
                "val_ema_mse_global": m_va.get("ema_mse_global", np.nan),
                "val_mixed": val_mixed,
            })
            print(f"{prefix}[Epoch {epoch:03d}] tr_loss={avg_tr_loss:.6f} | val_loss={avg_va_loss:.6f} | val_pearson={m_va['pearson']:.4f} | val_rmse={m_va['rmse']:.6f}")

    # 載回最佳權重
    model.load_state_dict(best_state)
    model.eval()

    # ---------- TEST ----------
    te_tgts, te_chunks = [], []
    test_loss_sum, test_n = 0.0, 0

    with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb_t = yb.to(device, non_blocking=True).long() if task_type == "classification" else yb.to(device, non_blocking=True).float()
            logits = model(Xb)

            if task_type == "classification":
                loss = loss_fn(logits, yb_t)
            else:
                loss = loss_fn(logits.squeeze(-1), yb_t)

            bs = Xb.size(0)
            test_loss_sum += loss.item() * bs
            test_n += bs

            if task_type == "classification":
                # 套「最佳 epoch」的溫度 best_T
                probs = F.softmax(logits / best_T, dim=1)
                te_chunks.append(probs.detach().cpu())
            else:
                te_chunks.append(logits.detach().float().cpu().squeeze(-1))
            te_tgts.append(yb_t.detach().cpu())

    avg_te_loss = test_loss_sum / max(1, test_n)
    result = {"history": history, "best_epoch": best_epoch, "state_dict": best_state}

    if task_type == "classification":
        # ======= 分類測試 =======
        y_te = torch.cat(te_tgts).numpy()
        y_prob_te = torch.cat(te_chunks).numpy()
        yhat_te = y_prob_te.argmax(axis=1)
        m_te = compute_metrics(y_te, yhat_te)
        print(f"{prefix}Test_acc={m_te['acc']:.3f} | test_f1={m_te['macro_f1']:.3f} | test_f05={m_te['f_05_macro']:.3f} | test_prec={m_te['macro_precision']:.3f} | test_re={m_te['macro_recall']:.3f} | mcc={m_te['mcc']:.3f}")

        best_thresh = None
        roc_info = None
        if num_class == 2:
            y_score = y_prob_te[:, 1]
            cfg_thresh = cfg["train"].get("threshold", None)
            if cfg_thresh is not None:
                best_thresh = float(cfg_thresh)
            elif best_val_thresh is not None:
                best_thresh = float(best_val_thresh)
            else:
                # ⚠️ 避免測試集門檻搜尋（資料外洩）；若要啟用，請在 cfg.train.allow_test_threshold_fallback=True
                if cfg["train"].get("allow_test_threshold_fallback", False):
                    best_thresh, roc_info = find_best_threshold_by_auc(y_te, y_score)
                else:
                    best_thresh = 0.5  # 後備：固定 0.5
            yhat_thresh = (y_score >= best_thresh).astype(int)
            m_thresh = compute_metrics(y_te, yhat_thresh)
        else:
            m_thresh = None

        result.update({
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
        })
        if num_class == 2:
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
            class_names=cfg["model"].get("class_names", None),  # None 時會自動以 C0..CN 命名
            save_dir=export_dir,
            prefix=f"fold_{fold_id}_",
            threshold=float(best_thresh) if (num_class == 2 and best_thresh is not None) else None
        )
        return model, result

    else:
        # ======= 回歸測試 =======
        y_te = torch.cat(te_tgts,  dim=0).numpy().astype(np.float64)      # ★
        y_pred_te = torch.cat(te_chunks, dim=0).numpy().astype(np.float64)   # ★
        m_te = compute_regression_metrics(y_te, y_pred_te)
        print(f"{prefix}Test_reg: pearson={m_te['pearson']:.4f} | rmse={m_te['rmse']:.6f} | mae={m_te['mae']:.6f}")

        result.update({
            "test_metrics_reg": {
                "test_loss": avg_te_loss, **m_te
            },
            "temperature": 1.0,
            "best_val_mixed": float(best_reg_mixed),
            "best_val_pearson": float(best_reg_pearson),
            "best_val_arrays": {  # 可選：方便外面分析
                "y_va": best_va_arrays[0].tolist() if best_va_arrays else None,
                "y_pred_va": best_va_arrays[1].tolist() if best_va_arrays else None
            }
        })

        # 回歸評估的繪圖（散點 / 殘差等）
        plot_regression_eval(
            y_true=y_te, y_pred=y_pred_te,
            save_dir=export_dir, prefix=f"fold_{fold_id}_"
        )

        # 若啟用回歸→分類門檻掃描，畫 Fβ vs threshold
        reg2cls = (cfg.get("regression_to_class", {}) or {})
        if reg2cls.get("enabled", False):
            plot_regression_threshold_sweep(
                y_true_reg=y_te, y_pred_reg=y_pred_te,
                true_threshold=float(reg2cls.get("true_threshold", 0.0)),
                beta=float(reg2cls.get("fbeta", 0.5)),
                grid_points=int(reg2cls.get("grid_points", 101)),
                save_dir=export_dir, prefix=f"fold_{fold_id}_"
            )

        # 匯出
        save_fold_metrics(history, save_dir=export_dir, prefix=f"fold_{fold_id}_")
        save_result(fold_id=fold_id, export_dir=export_dir, result=result)
        return model, result
