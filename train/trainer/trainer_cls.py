"""
trainer_cls.py
--------------
分類任務專用的訓練迴圈（single fold）。支援：
1) CE / Focal-CE 損失（含 class weight 與 label smoothing）
2) 類別分佈對齊（distribution alignment）與信心懲罰（confidence penalty）
3) 自動溫度校準（temperature scaling, CE 版）
4) 二分類動態門檻（threshold）搜尋：AUC-Youden 或 F-beta
5) 早停（可選用 val_loss 或 macro F0.5 作為主指標）
6) 訓練/驗證/測試的完整指標與可視化匯出

注意：
- 檔案假設在 CUDA/GPU 環境下執行（會 assert）
- 時序資料的資料載入器需保證 time-aware 的切分策略（外部保證）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from sklearn.metrics import fbeta_score

from .trainer_base import (
    get_task_type, amp_dtype, build_optimizer, build_warmup_scheduler, build_grad_scaler,
    infer_class_weights, fit_temperature_ce, find_best_threshold_by_auc
)

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 匯出/視覺化/度量
from train_utils.compute_export_metrices import (
    save_fold_metrics, plot_test_eval
)
from train_utils.metrics_cls import compute_cls_metrics  # ★ 新增（統一計算 acc/f1/f0.5/mcc 等）

# =========================================================
# 分類專屬 Loss（CrossEntropy / Focal CrossEntropy）
# =========================================================
class FocalLossCE(nn.Module):
    """
    1. 說明:
        Softmax 版 Focal Loss（針對多類別分類），可搭配 class weight。
        適用於類別不平衡或希望抑制「易分類樣本」的影響力。
        原始想法源自 Lin et al., 2017 (Focal Loss for Dense Object Detection)。

    2. inputs:
        gamma (float): Focal 指數 γ，越大表示越重懲罰易分類樣本（典型值 2.0）
        weight (Tensor|None): 類別權重（shape=[C]），常用於不平衡類別
        reduction (str): 'mean' 或 'sum'；建議 'mean'

    3. return:
        Callable / forward(logits, target) 回傳一個 scalar 的損失值 (Tensor)
    """
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, target):
        """
        1. 說明:
            計算 batch 的 Focal-CE 損失。先做 log_softmax，再套用 focal 調權。
        2. inputs:
            logits (FloatTensor): 模型輸出未經 softmax 的分數，shape = [B, C]
            target (LongTensor): 目標類別索引，shape = [B]
        3. return:
            loss (Tensor): 單一 scalar，預設為 mean
        """
        # 轉為對數機率
        logp = F.log_softmax(logits, dim=1)           # [B, C]
        p    = logp.exp()                             # [B, C]
        # focal 調權：對每一類別位置計算 -(1-p)^γ * log p
        loss = -((1 - p) ** self.gamma) * logp        # [B, C]
        # 只取等同於 target 類別的那一欄
        loss = loss.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]
        # 類別權重：只在對應的 target 位置乘上 weight
        if self.weight is not None:
            loss = loss * self.weight[target]
        # 聚合
        return loss.mean() if self.reduction == "mean" else loss.sum()


def build_classification_loss(cfg, class_weights):
    """
    1. 說明:
        依設定建立分類損失函數。支援:
        - CrossEntropyLoss（可含 label smoothing）
        - FocalLossCE（可含 class weight）
    2. inputs:
        cfg (dict): 設定檔（需包含 train 節點）
        class_weights (Tensor|None): 類別權重張量，shape=[C] 或 None
    3. return:
        loss_fn (nn.Module): 可呼叫的 loss 物件
    """
    ls = float(cfg["train"].get("label_smoothing", 0.0))
    use_focal = bool(cfg["train"].get("use_focal", False))
    if use_focal:
        return FocalLossCE(gamma=float(cfg["train"].get("focal_gamma", 2.0)),
                           weight=class_weights, reduction="mean")
    else:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)


# =========================================================
# 類別先驗估計與動態 Threshold 搜尋
# =========================================================
@torch.no_grad()
def infer_class_prior(train_loader, num_classes: int, device: str):
    """
    1. 說明:
        以 train_loader 的標籤頻率估計類別先驗分佈（class prior）。
        可搭配「分佈對齊懲罰」使用，避免模型崩成單一類。

    2. inputs:
        train_loader (DataLoader): 訓練資料載入器，需回傳 (X, y) 且 y 可展平為 [B]
        num_classes (int): 類別數
        device (str): 'cuda' 或 'cpu'（回傳張量會放在此裝置）

    3. return:
        prior (FloatTensor): 長度 = num_classes 的類別先驗，總和為 1
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, yb in train_loader:
        yb = yb.view(-1).cpu()
        binc = torch.bincount(yb, minlength=num_classes).to(torch.float64)
        counts[:num_classes] += binc[:num_classes]
    # 防呆：避免出現 0
    counts = counts.clamp(min=1)
    prior = (counts / counts.sum()).to(torch.float32).to(device)
    # 數值穩定
    prior = prior.clamp_min(1e-8)
    prior = prior / prior.sum()
    return prior


def find_best_threshold_by_fbeta(y_true, y_score_pos, *, beta: float = 0.5, grid_points: int = 201):
    """
    1. 說明:
        針對二分類，對正類機率 y_score_pos 用均勻網格掃描門檻 t∈[0,1]，
        以 F-beta（macro）作為目標找最佳 threshold。預設 beta=0.5（較重 Precision）。

    2. inputs:
        y_true (array-like): 0/1 真值，shape=[N]
        y_score_pos (array-like): 對「正類」的機率或分數，shape=[N]
        beta (float): F-beta 的 beta 參數，beta<1 偏重 Precision
        grid_points (int): 掃描網格點數（越大越細，但也越慢）

    3. return:
        best_thr (float): 最佳門檻
        info (dict): {"fbeta": 最佳Fbeta, "beta": beta}
    """
    y_true = np.asarray(y_true).astype(int)
    y_score_pos = np.asarray(y_score_pos).astype(float)
    thr_list = np.linspace(0.0, 1.0, int(grid_points))
    best_thr, best_val = 0.5, -1.0
    for t in thr_list:
        yhat = (y_score_pos >= t).astype(int)
        # macro: 對每個類別分別算 fbeta，再平均；若單類導致除 0，zero_division=0
        val = fbeta_score(y_true, yhat, beta=beta, average="macro", zero_division=0)
        if val > best_val:
            best_val, best_thr = val, t
    return float(best_thr), {"fbeta": float(best_val), "beta": float(beta)}


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
        - 訓練：支援 AMP、warmup scheduler、梯度裁剪、分佈對齊與信心懲罰。
        - 驗證：計算 CE 損失與分類指標，溫度校準，二分類時可掃描門檻（AUC 或 F-beta）。
        - 早停：以主指標（val_loss 或 macro F0.5）決定是否更新最佳模型。
        - 測試：載回最佳權重與最佳溫度，產出最終測試指標與可視化。

    2. inputs:
        model (nn.Module): 分類模型，forward(X)->logits [B,C]
        train_loader/val_loader/test_loader (DataLoader): 對應資料載入器
        cfg (dict): 設定檔（含 model/train/objective 等欄位）
        device (str|None): 預設 'cuda'
        fold_id (int|None): 目前 fold 編號（僅用於列印/檔名）
        export_dir (str|None): 匯出歷史指標與圖表的目錄

    3. return:
        model (nn.Module): 已載入最佳權重的模型
        result (dict): 包含 history/best_epoch/val_metrics/test_metrics/
                       best_val_thresh/temperature/threshold_metrics 等資訊
    """
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

    # ---- 準備優化器 / 語意精度 / AMP scaler / scheduler ----
    model = model.to(device)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_warmup_scheduler(optimizer, steps_per_epoch, cfg)
    dtype = amp_dtype(cfg=cfg)                # 例如 torch.float16 或 bfloat16，或 None（停用 AMP）
    scaler = build_grad_scaler(dtype)

    # ---- 檢查輸出維度並建立 loss ----
    model.eval()
    with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=(dtype is not None)):
        xb0, _ = next(iter(train_loader))
        xb0 = xb0[:1].to(device, non_blocking=True)
        logits0 = model(xb0)
        out_dim = int(logits0.shape[-1])
    model.train()
    if out_dim != num_class:
        raise ValueError(
            f"[trainer_cls] 模型 out_dim={out_dim} != num_classes={num_class}。請調整 head。"
        )
    # 依訓練集標籤頻率推得 class weights（不平衡時有幫助）
    class_weights = infer_class_weights(train_loader, num_class, device)
    loss_fn = build_classification_loss(cfg, class_weights=class_weights)

    # ---- 額外正則項開關 ----
    # 分佈對齊（將 batch 平均預測分佈拉近先驗）
    dist_align_weight = float(cfg["train"].get("dist_align_weight", 0.0))
    # 信心懲罰（負熵）：抑制過度自信，避免全部壓成同一類
    conf_penalty_w    = float(cfg["train"].get("confidence_penalty", 0.0))
    prior_mode        = str(cfg["train"].get("dist_prior", "train")).lower()
    if dist_align_weight > 0.0:
        if prior_mode == "uniform":
            class_prior = torch.full((num_class,), 1.0 / max(1, num_class), dtype=torch.float32, device=device)
        else:
            class_prior = infer_class_prior(train_loader, num_class, device)
    else:
        class_prior = None

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

        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()

            if not printed_shape:
                print("[trainer_cls] DEBUG batch X shape =", tuple(Xb.shape))
                printed_shape = True

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", dtype=dtype, enabled=(dtype is not None)):
                logits = model(Xb)
                loss = loss_fn(logits, yb)

                # (1) 分佈對齊：KL( mean(p) || prior )
                if dist_align_weight > 0.0 and class_prior is not None and logits.shape[-1] > 1:
                    probs = torch.softmax(logits, dim=1).float()  # [B, C]
                    p_mean = probs.mean(dim=0)                     # [C]
                    kl = F.kl_div((p_mean + 1e-8).log(), class_prior, reduction="batchmean")
                    loss = loss + dist_align_weight * kl

                # (2) 信心懲罰：期望的負熵（越自信惡化，鼓勵保守）
                if conf_penalty_w > 0.0 and logits.shape[-1] > 1:
                    probs = torch.softmax(logits, dim=1).float()
                    conf_loss = (probs * (probs + 1e-8).log()).sum(dim=1).mean()  # = -Entropy 的期望值
                    loss = loss + conf_penalty_w * conf_loss

            # 反傳 + AMP 梯度縮放
            scaler.scale(loss).backward()
            if clip and clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

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

        with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=(dtype is not None)):
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).long()
                logits = model(Xb)
                loss = loss_fn(logits, yb)

                bs = Xb.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs
                val_tgts.append(yb.detach().cpu())
                val_logits.append(logits.detach())  # 保持 GPU 張量，後續做溫度校準較方便

        avg_va_loss = val_loss_sum / max(1, val_n)
        y_va = torch.cat(val_tgts, dim=0).numpy()
        logits_all = torch.cat(val_logits, dim=0)  # [N, C]

        # ---- 溫度校準：以驗證集 CE 最佳化找到該 epoch 的 T ----
        if cfg["train"].get("use_temperature", True):
            # 注意：fit_temperature_ce 會在 device 上計算
            T_epoch = fit_temperature_ce(logits_all, torch.from_numpy(y_va).to(device))
        else:
            T_epoch = torch.tensor(1.0, device=device)

        # 校準後機率（對多類別直接 softmax）
        probs_all = torch.softmax((logits_all / T_epoch).float(), dim=1).cpu().numpy()

        # ---- 二分類：嘗試找到更合理的驗證門檻（多類別直接 argmax）----
        if probs_all.shape[1] == 2:
            y_score_va = probs_all[:, 1]  # 正類分數
            thr_mode = str(cfg["train"].get("threshold_mode", "auto_auc")).lower()
            if thr_mode == "auto_fbeta":
                beta = float(cfg["train"].get("threshold_fbeta", 0.5))
                grid_points = int(cfg["train"].get("threshold_grid_points", 201))
                curr_val_thresh, _ = find_best_threshold_by_fbeta(y_va, y_score_va, beta=beta, grid_points=grid_points)
            elif thr_mode == "fixed":
                cfg_thresh = cfg["train"].get("threshold", None)
                curr_val_thresh = float(cfg_thresh) if cfg_thresh is not None else 0.5
            else:  # 預設 auto_auc（Youden J from ROC）
                curr_val_thresh, _ = find_best_threshold_by_auc(y_va, y_score_va)
            yhat_va = (y_score_va >= curr_val_thresh).astype(int)
        else:
            yhat_va = probs_all.argmax(axis=1)
            curr_val_thresh = None

        m_va = compute_cls_metrics(y_va, yhat_va)

        # ---- Early Stopping：主指標為 macro F0.5 或 val_loss ----
        if primary_is_f05:
            improved = (m_va.get("macro_f05", m_va.get("f_05_macro", -1.0)) > (best_val_f_05 + 1e-6))
        else:
            improved = (avg_va_loss < (best_cls_val_loss - 1e-6))

        if epoch == 1 or improved:
            # 更新最佳狀態
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

        # ---- 記錄歷史（供圖表與 CSV 匯出）----
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

    # ---- 載回最佳權重 ----
    model.load_state_dict(best_state)
    model.eval()

    # -------- TEST --------
    te_tgts, te_probs = [], []
    test_loss_sum, test_n = 0.0, 0

    with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=(dtype is not None)):
        for Xb, yb in test_loader:
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

    avg_te_loss = test_loss_sum / max(1, test_n)

    y_te = torch.cat(te_tgts).numpy()
    y_prob_te = torch.cat(te_probs).float().numpy() 
    yhat_te = y_prob_te.argmax(axis=1)

    # --- 印出前 10 筆測試樣本的預測機率（方便肉眼 sanity check）---
    class_names = cfg["model"].get("class_names", [str(i) for i in range(y_prob_te.shape[1])])
    print("\n[Test Predictions - top 10]")
    for i in range(min(10, len(y_prob_te))):
        probs = y_prob_te[i]
        prob_str = ", ".join([f"{c}: {p:.4f}" for c, p in zip(class_names, probs)])
        print(f"[{i}] True: {y_te[i]}, Pred: {yhat_te[i]}, Probs: [{prob_str}]")

    # ---- 測試指標（使用 argmax 產生的 yhat_te）----
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
    
    # ---- 二分類的 threshold 報表（避免在測試集上搜尋，以驗證最佳為主）----
    best_thresh, m_thresh = None, None
    is_binary_test = (y_prob_te.shape[1] == 2)

    if is_binary_test:
        y_score = y_prob_te[:, 1]  # 正類分數
        thr_mode = str(cfg["train"].get("threshold_mode", "auto_auc")).lower()
        if thr_mode == "fixed":
            cfg_thresh = cfg["train"].get("threshold", None)
            best_thresh = float(cfg_thresh) if cfg_thresh is not None else (float(best_val_thresh) if best_val_thresh is not None else 0.5)
        elif best_val_thresh is not None:
            best_thresh = float(best_val_thresh)
        else:
            # 不在測試集上尋找門檻，避免資料外洩；無驗證門檻時回退 0.5
            best_thresh = 0.5

        yhat_thresh = (y_score >= best_thresh).astype(int)
        m_thresh = compute_cls_metrics(y_te, yhat_thresh)

    # ---- 組合回傳結果 ----
    result = {
        "history": history,
        "best_epoch": best_epoch,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "val_metrics":{
            "val_loss": float(best_cls_val_loss),
            "macro_f1": float(best_val_f1),
            "f_05_macro": float(best_val_f_05),
            "macro_precision": float(best_val_prec),
            "macro_recall": float(best_val_recall),},

        "test_metrics": {    
            "test_acc":        m_te.get("acc", 0.0),
            "test_macro_f1":   m_te.get("macro_f1", 0.0),
            "test_weighted_f1":m_te.get("weighted_f1", m_te.get("macro_f1", 0.0)),
            "test_macro_f05":  m_te.get("macro_f05", m_te.get("f_05_macro", 0.0)),
            "test_mcc":        m_te.get("mcc", 0.0),
        },

        "best_val_thresh": float(best_val_thresh) if best_val_thresh is not None else None,
        "temperature": float(best_T.item()) if torch.is_tensor(best_T) else 1.0,

        "threshold_metrics":{
            "best_val_thresh": float(best_val_thresh) if best_val_thresh is not None else None,
            "temperature": float(best_T.item()) if torch.is_tensor(best_T) else 1.0,
            }
        }
    if is_binary_test and m_thresh is not None:
        # 提供在「固定最佳門檻」下的整體測試指標（通常比 argmax 更貼近交易決策）
        result["threshold_metrics"] = {
            "best_threshold": float(best_thresh),
            "acc": m_thresh["acc"],
            "macro_f1": m_thresh["macro_f1"],
            "macro_precision": m_thresh["macro_precision"],
            "macro_recall": m_thresh["macro_recall"],
            "macro_f05": m_thresh.get("macro_f05", m_thresh.get("f_05_macro", 0.0)),
        }

    # ---- 匯出與作圖（歷史 CSV / ROC-PR / 混淆矩陣等）----
    save_fold_metrics(history, save_dir=export_dir, prefix=f"fold_{fold_id}_")
    plot_test_eval(
        y_true=y_te, y_pred=yhat_te, y_prob=y_prob_te,
        class_names=cfg["model"].get("class_names", None),
        save_dir=export_dir,
        prefix=f"fold_{fold_id}_",
        threshold=float(best_thresh) if (y_prob_te.shape[1] == 2 and best_thresh is not None) else None
    )
    return model, result
