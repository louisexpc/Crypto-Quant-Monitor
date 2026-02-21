# train/training/trainers/classification.py
"""
--------------
分類任務專用的訓練迴圈（single fold）。支援：
1) CrossEntropy 損失（可選 class weight）
2) 自動溫度校準（temperature scaling, CE 版）
3) 二分類動態門檻（threshold）搜尋：fixed 或 auto_fbeta
4) 早停（可選用 val_loss 或 macro F0.5 作為主指標）
5) 訓練/驗證/測試的完整指標與可視化匯出
6) CollapseGuard：PPR/熵 監控與自救（λ_cp 調整、LR 衰減、可選回滾最佳權重）

注意：
- 檔案假設在 CUDA/GPU 環境下執行（會 assert）
- 時序資料的資料載入器需保證 time-aware 的切分策略（外部保證）
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from collections import Counter
from train.training.trainers.utils import (
    build_optimizer,
    build_warmup_scheduler,
    infer_class_weights,
    _iter_batches,
    find_best_threshold_by_fbeta,
    fit_temperature_ce
    )

from train.training.metrics.metrics_cls import compute_cls_metrics
from train.training.losses.cls import build_classification_loss
from train.models.xgb_model import XGBClassifierModel
from train.training.trainers.xgb import _train_one_fold_xgb

# =========================================================
# 主訓練流程（單一 fold）
# =========================================================
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
    """
    1. 說明:
        針對單一 fold 執行完整的訓練/驗證/測試流程。
        - 訓練：支援 warmup、梯度裁剪。
        - 驗證：CE 指標、溫度校準、二分類門檻搜尋。
        - 早停：以主指標（val_loss 或 macro F0.5）決定最佳模型。
        - 測試：載回最佳權重與最佳溫度，輸出指標與圖表。
        - CollapseGuard：PPR/熵 監控，自動調 λ_cp 與 LR，必要時回滾最佳權重。
    2. inputs:
        model (nn.Module): forward(X)->logits [B,C]
        *_loader (DataLoader): 三段資料
        cfg (dict): 設定檔
        device (str|None): 預設 'cuda'
        fold_id (int|None): fold 編號
        export_dir (str|None): 匯出路徑
    3. return:
        model (nn.Module), result (dict)
    """
    if XGBClassifierModel is not None and isinstance(model, XGBClassifierModel):
        return _train_one_fold_xgb(model, cfg, fold_id, export_dir)

    if len(train_loader) == 0:
        print("[ERROR][trainer_cls] empty train loader")
        return None, None

    # 僅允許 GPU
    assert torch.cuda.is_available(), "[trainer_cls] 需要 CUDA GPU 環境。"
    device = device or "cuda"

    # ---- 讀取訓練設定 ----
    lr        = float(cfg["train"]["lr"])
    clip      = float(cfg["train"]["grad_clip"])
    epochs    = int(cfg["train"]["epochs"])
    patience  = int(cfg["train"]["early_stopping_patience"])
    num_class = int(cfg["model"]["num_classes"])
    primary_metric = str(cfg.get("objective", {}).get("primary_metric", "val_loss")).lower()
    primary_is_f05 = primary_metric in ["macro_f05", "f_05_macro", "threshold_macro_f05"]

    # ---- 準備優化器 / scheduler ----
    model = model.to(device)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_warmup_scheduler(optimizer, steps_per_epoch, cfg)

    # ---- 檢查輸出維度並建立 loss ----
    model.eval()
    with torch.no_grad():
        xb0, _ = next(iter(train_loader))
        xb0 = xb0[:1].to(device, non_blocking=True)
        logits0 = model(xb0)
        out_dim = int(logits0.shape[-1])
    model.train()
    if out_dim != num_class:
        raise ValueError(f"[trainer_cls] 模型 out_dim={out_dim} != num_classes={num_class}。請調整 head。")

    # class weights
    class_weights = infer_class_weights(train_loader, num_class, device)
    loss_fn = build_classification_loss(cfg, class_weights=class_weights)


    # ---- Early-stop 狀態 ----
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
    best_T = torch.tensor(1.0, device=device)  # 溫度校準參數（每個 fold 最佳）

    printed_shape = False

    # =========================
    #         EPOCH 迴圈
    # =========================
    for epoch in range(1, epochs + 1):
        # -------- TRAIN --------
        model.train()
        train_loss_sum, train_n = 0.0, 0
        tr_preds, tr_tgts = [], []

        for Xb, yb in _iter_batches(train_loader, device, int(cfg["train"]["batch_size"])):
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()

            if not printed_shape:
                print("[trainer_cls] DEBUG batch X shape =", tuple(Xb.shape))
                printed_shape = True

            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss = loss_fn(logits, yb)

            loss.backward()
            if clip and clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()

            # 累計 loss 與收集訓練預測（for train 指標）
            bs = Xb.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

            preds = logits.argmax(dim=-1)
            tr_preds.append(preds.detach().cpu())
            tr_tgts.append(yb.detach().cpu())

        avg_tr_loss = train_loss_sum / max(1, train_n)
        y_tr = torch.cat(tr_tgts).numpy()
        yhat_tr = torch.cat(tr_preds).numpy()
        m_tr = compute_cls_metrics(y_tr, yhat_tr)

        # -------- VAL --------
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        val_tgts = []
        val_logits = []

        with torch.no_grad():
            for Xb, yb in _iter_batches(val_loader, device, int(cfg["train"]["batch_size"])):
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()
                logits = model(Xb)
                loss = loss_fn(logits, yb)

                bs = Xb.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs
                val_tgts.append(yb.detach().cpu())
                val_logits.append(logits.detach())

        avg_va_loss = val_loss_sum / max(1, val_n)
        y_va = torch.cat(val_tgts, dim=0).numpy()
        logits_all = torch.cat(val_logits, dim=0)  # [N, C]

        # ---- 溫度校準 ----
        if cfg["train"].get("use_temperature", True):
            T_epoch = fit_temperature_ce(logits_all, torch.from_numpy(y_va).to(device))
        else:
            T_epoch = torch.tensor(1.0, device=device)

        probs_all = torch.softmax((logits_all / T_epoch).float(), dim=1).cpu().numpy()

        # ---- 二分類：驗證門檻 ----
        if probs_all.shape[1] == 2:
            y_score_va = probs_all[:, 1]
            thr_mode = str(cfg["train"].get("threshold_mode", "auto_auc")).lower()
            if thr_mode == "auto_fbeta":
                beta = float(cfg["train"].get("threshold_fbeta", 0.5))
                grid_points = int(cfg["train"].get("threshold_grid_points", 201))
                curr_val_thresh, _ = find_best_threshold_by_fbeta(y_va, y_score_va, beta=beta, grid_points=grid_points)
            else:                
                cfg_thresh = cfg["train"].get("threshold", None)
                curr_val_thresh = float(cfg_thresh) if cfg_thresh is not None else 0.5

            yhat_va = (y_score_va >= curr_val_thresh).astype(int)
        else:
            yhat_va = probs_all.argmax(axis=1)
            curr_val_thresh = None

        m_va = compute_cls_metrics(y_va, yhat_va)

        # ---- Early Stopping ----
        improved = (avg_va_loss < (best_cls_val_loss - 1e-6)) if not primary_is_f05 else \
                   (m_va.get("macro_f05", m_va.get("f_05_macro", -1.0)) > (best_val_f_05 + 1e-6))

        if epoch == 1 or improved:
            best_val_f1 = m_va.get("macro_f1", best_val_f1)
            best_val_f_05 = m_va.get("macro_f05", m_va.get("f_05_macro", best_val_f_05))
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

        # ---- 記錄歷史 ----
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
        print(f"{prefix}[Epoch {epoch:03d}] tr_loss={avg_tr_loss:.4f} | val_loss={avg_va_loss:.4f} | "
              f"val_acc={m_va.get('acc',np.nan):.3f} | val_f1={m_va.get('macro_f1',np.nan):.3f} | "
              f"val_f05={m_va.get('f_05_macro',np.nan):.3f}")

    # ---- 載回最佳權重 ----
    model.load_state_dict(best_state)
    model.eval()

    # -------- TEST --------
    te_tgts, te_probs = [], []
    test_loss_sum, test_n = 0.0, 0

    with torch.no_grad():
        for Xb, yb in _iter_batches(test_loader, device, int(cfg["train"]["batch_size"])):
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()
            logits = model(Xb)
            loss = loss_fn(logits, yb)

            bs = Xb.size(0)
            test_loss_sum += loss.item() * bs
            test_n += bs

            # 用「最佳溫度」做校準，再取 softmax 機率
            probs = torch.softmax((logits / best_T).float(), dim=1).float()
            te_probs.append(probs.detach().cpu())
            te_tgts.append(yb.detach().cpu())

    y_te = torch.cat(te_tgts).numpy()
    y_prob_te = torch.cat(te_probs).float().numpy()

    # --- 二分類：測試只用驗證最佳門檻 ---
    is_binary_test = (y_prob_te.shape[1] == 2)
    best_thresh = None
    if is_binary_test:
        y_score = y_prob_te[:, 1]

        # 以驗證最佳門檻為主；若沒學到（理論上會有），再依 cfg 或 0.5
        thr_mode = str(cfg["train"].get("threshold_mode", "auto_auc")).lower()
        if thr_mode == "fixed":
            cfg_thresh = cfg["train"].get("threshold", None)
            best_thresh = float(cfg_thresh) if cfg_thresh is not None else (
                float(best_val_thresh) if best_val_thresh is not None else 0.5
            )
        elif best_val_thresh is not None:
            best_thresh = float(best_val_thresh)
        else:
            best_thresh = 0.5  # 極端退避

        yhat_te = (y_score >= best_thresh).astype(int)

        # 列印 Top-10：顯示用 threshold 的預測
        class_names = cfg["model"].get("class_names", ["neg", "pos"])
        print("\n[Test Predictions - top 10 @thr={:.3f}]".format(best_thresh))
        for i in range(min(10, len(y_prob_te))):
            probs = y_prob_te[i]
            prob_str = ", ".join([f"{c}: {p:.4f}" for c, p in zip(class_names, probs)])
            pred_thr = int(yhat_te[i])
            print(f"[{i}] True: {y_te[i]}, Pred_thr: {pred_thr}, Probs: [{prob_str}]")

        # 用 threshold 的結果作為「唯一的」測試指標
        m_te = compute_cls_metrics(y_te, yhat_te)
        test_f05 = m_te.get("macro_f05", m_te.get("f_05_macro", 0.0))
        print(
            f"{prefix}Test@thr={best_thresh:.3f} | "
            f"acc={m_te.get('acc', 0.0):.3f} | "
            f"f1={m_te.get('macro_f1', 0.0):.3f} | "
            f"f05={test_f05:.3f} | "
            f"prec={m_te.get('macro_precision', 0.0):.3f} | "
            f"rec={m_te.get('macro_recall', 0.0):.3f} | "
            f"mcc={m_te.get('mcc', 0.0):.3f}"
        )

    else:
        # 多分類→維持 argmax
        yhat_te = y_prob_te.argmax(axis=1)
        class_names = cfg["model"].get("class_names", [str(i) for i in range(y_prob_te.shape[1])])
        print("\n[Test Predictions - top 10]")
        for i in range(min(10, len(y_prob_te))):
            probs = y_prob_te[i]
            prob_str = ", ".join([f"{c}: {p:.4f}" for c, p in zip(class_names, probs)])
            print(f"[{i}] True: {y_te[i]}, Pred: {yhat_te[i]}, Probs: [{prob_str}]")

        m_te = compute_cls_metrics(y_te, yhat_te)
        test_f05 = m_te.get("macro_f05", m_te.get("f_05_macro", 0.0))
        print(
            f"{prefix}Test_acc={m_te.get('acc', 0.0):.3f} | "
            f"test_f1={m_te.get('macro_f1', 0.0):.3f} | "
            f"test_f05={test_f05:.3f} | "
            f"test_prec={m_te.get('macro_precision', 0.0):.3f} | "
            f"test_re={m_te.get('macro_recall', 0.0):.3f} | "
            f"mcc={m_te.get('mcc', 0.0):.3f}"
        )



    # 1) 做出 loss 的歷史序列（collapse_mask 會拿這個用）
    val_loss_history = [float(h.get("val_loss", np.nan)) for h in history if "val_loss" in h]
    train_loss_history = [float(h.get("train_loss", np.nan)) for h in history if "train_loss" in h]

    # 2) 標籤分佈（collapse_mask 算 pos_ratio 用）
    lc_tr = dict(Counter(y_tr.tolist()))  # 最後一個 epoch 的 train 走完整個資料集，分佈即為全訓練集
    lc_va = dict(Counter(y_va.tolist()))  # 驗證集同理
    label_counts = {"train": lc_tr, "val": lc_va}

    # ---- 組合回傳 ----
    result = {
        "history": history,
        "val_loss_history": val_loss_history,
        "label_counts": label_counts, 
        "best_epoch": best_epoch,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "val_metrics": {
            "val_loss": float(best_cls_val_loss),
            "macro_f1": float(best_val_f1),
            "f_05_macro": float(best_val_f_05),
            "macro_precision": float(best_val_prec),
            "macro_recall": float(best_val_recall),
        },
        # ★ test_metrics：二分類時就是用 best_val_thresh 的那組
        "test_metrics": {
            "test_acc":        m_te.get("acc", 0.0),
            "test_macro_f1":   m_te.get("macro_f1", 0.0),
            "test_weighted_f1":m_te.get("weighted_f1", m_te.get("macro_f1", 0.0)),
            "test_macro_f05":  m_te.get("macro_f05", m_te.get("f_05_macro", 0.0)),
            "test_mcc":        m_te.get("mcc", 0.0),
        },
        "best_val_thresh": float(best_val_thresh) if best_val_thresh is not None else None,
        "temperature": float(best_T.item()) if torch.is_tensor(best_T) else 1.0,
        "threshold_metrics": {}  # 下面填
    }

    # threshold_metrics 與圖的 threshold 也跟 test 使用的邏輯一致
    if is_binary_test:
        result["threshold_metrics"].update({
            "best_threshold": float(best_thresh),
            "acc": result["test_metrics"]["test_acc"],
            "macro_f1": result["test_metrics"]["test_macro_f1"],
            "macro_precision": result["test_metrics"]["test_macro_precision"] if "test_macro_precision" in result["test_metrics"] else m_te.get("macro_precision", 0.0),
            "macro_recall": result["test_metrics"]["test_macro_recall"] if "test_macro_recall" in result["test_metrics"] else m_te.get("macro_recall", 0.0),
            "macro_f05": result["test_metrics"]["test_macro_f05"],
        })
    else:
        result["threshold_metrics"] = {
            "best_threshold": None,
            "acc": result["test_metrics"]["test_acc"],
            "macro_f1": result["test_metrics"]["test_macro_f1"],
            "macro_precision": m_te.get("macro_precision", 0.0),
            "macro_recall": m_te.get("macro_recall", 0.0),
            "macro_f05": result["test_metrics"]["test_macro_f05"],
        }

    # ---- 匯出與作圖 ----
    result["eval_payload"] = {
        "y_true": y_te,
        "y_pred": yhat_te,
        "y_prob": y_prob_te,
        "class_names": cfg["model"].get("class_names", None),
        "best_threshold": float(best_thresh) if (is_binary_test and best_thresh is not None) else None,
    }
    return model, result
