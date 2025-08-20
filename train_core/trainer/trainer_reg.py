# trainer_reg.py
import numpy as np
import torch
import torch.nn as nn
from torch import amp

from .trainer_base import (
    amp_dtype, build_optimizer, build_warmup_scheduler, build_grad_scaler
)
from .xgb_trainer import _train_one_fold_xgb

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 回歸度量/損失/圖表
from utils.regression_utils import compute_regression_metrics, build_regression_loss
from utils.compute_export_metrices import (
    save_fold_metrics, save_result,
    plot_regression_eval, plot_regression_threshold_sweep,
    compute_mixed_objective_np
)
from models.xgb_model import XGBRegressorModel

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
    
    # === XGB 分支：直接改走 numpy 訓練，略過整個 torch 流程 ===
    if XGBRegressorModel is not None and isinstance(model, XGBRegressorModel):
        return _train_one_fold_xgb(model, cfg, fold_id, export_dir)
    

    # === DL: PyTorch 訓練迴圈 ===
    if len(train_loader) == 0:
        print("[ERROR][trainer_reg] empty train loader")
        return None, None

    assert torch.cuda.is_available(), "[trainer_reg] 需要 CUDA GPU 環境。"
    device = device or "cuda"

    # 讀設定
    lr        = float(cfg["train"]["lr"])
    clip      = float(cfg["train"]["grad_clip"])
    epochs    = int(cfg["train"]["epochs"])
    patience  = int(cfg["train"]["early_stopping_patience"])
    primary   = (cfg.get("task", {}) or {}).get("primary_metric", "val_loss").lower()

    model = model.to(device)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_warmup_scheduler(optimizer, steps_per_epoch, cfg)
    dtype = amp_dtype()
    scaler = build_grad_scaler(dtype)

    # 回歸損失（由 cfg 決定：mse/huber 或 混合 α·EMA-MSE + β·(1-Pearson) 等）
    loss_fn = build_regression_loss(cfg)

    # Early-stop 狀態
    best_epoch = 0
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    wait = 0
    prefix = f"[fold {fold_id}] " if fold_id is not None else ""
    history = []

    best_reg_val_loss = float("inf")
    best_reg_pearson  = -1.0
    best_reg_mixed    = float("inf")
    best_va_arrays    = None  # (y_va, y_pred_va)

    printed_shape = False

    # ----------------- Epoch Loop -----------------
    for epoch in range(1, epochs + 1):
        # -------- TRAIN --------
        model.train()
        train_loss_sum, train_n = 0.0, 0
        tr_preds, tr_tgts = [], []

        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()

            if not printed_shape:
                print("[trainer_reg] DEBUG batch X shape =", tuple(Xb.shape))
                printed_shape = True

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
                pred = model(Xb).squeeze(-1)
                loss = loss_fn(pred, yb)

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

            tr_preds.append(pred.detach().float().cpu())
            tr_tgts.append(yb.detach().cpu())

        avg_tr_loss = train_loss_sum / max(1, train_n)
        y_tr = torch.cat(tr_tgts).numpy()
        yhat_tr = torch.cat(tr_preds).numpy()
        m_tr = compute_regression_metrics(y_tr, yhat_tr)

        # -------- VAL --------
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        val_tgts = []
        val_preds = []

        with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()
                pred = model(Xb).squeeze(-1)
                loss = loss_fn(pred, yb)

                bs = Xb.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs
                val_tgts.append(yb.detach().cpu())
                val_preds.append(pred.detach().float().cpu())

        avg_va_loss = val_loss_sum / max(1, val_n)
        y_va = torch.cat(val_tgts, dim=0).to(torch.float32).cpu().numpy()
        y_pred_va = torch.cat(val_preds, dim=0).to(torch.float32).cpu().numpy()

        m_va = compute_regression_metrics(y_va, y_pred_va)

        # 組合指標（Optuna 主目標 / Early-stop 也可用）
        alpha = float(cfg["loss"].get("alpha", 0.7))
        beta  = float(cfg["loss"].get("beta", 0.3))
        decay = float(cfg["loss"].get("ema_decay", 0.9))
        val_mixed, comps = compute_mixed_objective_np(
            y_true=y_va, y_pred=y_pred_va, alpha=alpha, beta=beta, ema_decay=decay
        )
        m_va["pearson_global"] = comps["pearson"]
        m_va["ema_mse_global"] = comps["ema_mse"]
        m_va["mixed"]          = val_mixed

        if epoch == 1:
            std_pred = float(np.std(y_pred_va))
            r_sanity = float(np.corrcoef(y_va, y_pred_va)[0,1]) if len(y_va) > 2 else 0.0
            print(f"[trainer_reg][DEBUG] val std(y_pred)={std_pred:.6g}, corr={r_sanity:.4f}")

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

    # 載回最佳
    model.load_state_dict(best_state)
    model.eval()

    # -------- TEST --------
    te_tgts, te_preds = [], []
    test_loss_sum, test_n = 0.0, 0

    with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=True):
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()
            pred = model(Xb).squeeze(-1)
            loss = loss_fn(pred, yb)

            bs = Xb.size(0)
            test_loss_sum += loss.item() * bs
            test_n += bs

            te_preds.append(pred.detach().float().cpu())
            te_tgts.append(yb.detach().cpu())

    avg_te_loss = test_loss_sum / max(1, test_n)
    y_te = torch.cat(te_tgts,  dim=0).numpy().astype(np.float64)
    y_pred_te = torch.cat(te_preds, dim=0).numpy().astype(np.float64)
    m_te = compute_regression_metrics(y_te, y_pred_te)
    print(f"{prefix}Test_reg: pearson={m_te['pearson']:.4f} | rmse={m_te['rmse']:.6f} | mae={m_te['mae']:.6f}\n")

    result = {
        "history": history,
        "best_epoch": best_epoch,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "test_metrics_reg": {
            "test_loss": avg_te_loss, **m_te
        },
        "temperature": 1.0,
        "best_val_mixed": float(best_reg_mixed),
        "best_val_pearson": float(best_reg_pearson),
        "best_val_arrays": {
            "y_va": best_va_arrays[0].tolist() if best_va_arrays else None,
            "y_pred_va": best_va_arrays[1].tolist() if best_va_arrays else None
        }
    }

    # 可選：回歸→分類門檻掃描
    reg2cls = cfg["regression_to_class"]
    if reg2cls["enabled"]:
        plot_regression_threshold_sweep(
            y_true_reg=y_te, y_pred_reg=y_pred_te,
            true_threshold=float(reg2cls.get("true_threshold", 0.0)),
            beta=float(reg2cls.get("fbeta", 0.5)),
            grid_points=int(reg2cls.get("grid_points", 101)),
            save_dir=export_dir, prefix=f"fold_{fold_id}_"
        )

    # 繪圖 / 匯出
    plot_regression_eval(
        y_true=y_te, y_pred=y_pred_te,
        save_dir=export_dir, prefix=f"fold_{fold_id}_"
    )
    save_fold_metrics(history, save_dir=export_dir, prefix=f"fold_{fold_id}_")
    save_result(fold_id=fold_id, export_dir=export_dir, result=result)
    return model, result
