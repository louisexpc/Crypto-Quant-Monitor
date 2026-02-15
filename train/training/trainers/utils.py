# train/training/trainers/utils.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, Callable
import math
import numpy as np
import torch
import torch.nn as nn
from torch import amp
from sklearn.metrics import fbeta_score
from torch.optim.lr_scheduler import LambdaLR

__all__ = [
    "get_task_type", "get_trainer",
    "amp_dtype", "build_optimizer", "build_warmup_scheduler", "build_grad_scaler",
    "infer_class_weights", "infer_class_prior",
    "find_best_threshold_by_fbeta",
    "_iter_batches",
]


# -----------------------------
# 小工具：把可能是 list/tuple 的數值轉成純 float
# -----------------------------
def _as_float(x, name="value", default=None):
    """
    1. 說明:
        將輸入數值（或列表/元組）轉為單一 float；若給的是序列，取第 1 個元素。
    2. inputs:
        - x: int|float|list|tuple，可轉為浮點的來源值。
        - name (str): 用於錯誤訊息的欄位名。
        - default (float|None): 當 x 不可判讀時的回退值（None 表示拋例外）。
    3. return:
        - v (float): 轉換後的浮點數。
    """
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return float(x[0])
    if default is not None:
        return float(default)
    raise TypeError(f"[trainer_base] {name} expects a float or list/tuple, got {type(x)}")

# =========================================================
# GPU 預載資料集直切
# =========================================================
def _iter_batches(loader, device: str, batch_size: int):
    """
    1. 說明:
        若 Dataset 已將完整資料預載為 GPU/CUDA 張量（具 X/y tensor，且在目標裝置上），
        以連續切片方式產生 batch，避免 DataLoader 的堆疊/小 kernel 開銷；否則回退到一般迭代。
    2. inputs:
        - loader (DataLoader): PyTorch DataLoader。
        - device (str): 目標裝置類型（'cuda' 或 'cpu'）。
        - batch_size (int): 每批大小。
    3. return:
        - 產生器：逐批回傳 (Xb, yb)。
    """
    try:
        ds = getattr(loader, "dataset", None)
        if ds is not None and hasattr(ds, "X") and hasattr(ds, "y"):
            X, y = ds.X, ds.y
            if isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):
                if X.device.type == device and y.device.type == device:
                    n = int(y.shape[0])
                    bs = max(1, int(batch_size))
                    for s in range(0, n, bs):
                        e = min(n, s + bs)
                        yield X[s:e], y[s:e]
                    return
    except Exception:
        pass
    for xb, yb in loader:
        yield xb, yb

# -----------------------------
# 任務/裝置/AMP dtype 公用
# -----------------------------
def get_task_type(cfg: dict) -> str:
    """
    1. 說明:
        由設定判斷任務型態：若 num_classes>=2 則為分類，否則回歸。
    2. inputs:
        - cfg (dict): 設定檔（可含 task.type / target.type / model.num_classes）。
    3. return:
        - task (str): 'classification' 或 'regression'。
    """
    if "task" in cfg and "type" in cfg["task"]:
        return str(cfg["task"]["type"]).lower()
    if "target" in cfg and "type" in cfg["target"]:
        return str(cfg["target"]["type"]).lower()
    nc = int(cfg["model"].get("num_classes", 1))
    return "classification" if nc >= 2 else "regression"

def amp_dtype(cfg: dict | None = None):
    """
    1. 說明:
        根據裝置支援與設定，選擇 AMP 計算 dtype：bf16 > fp16；或關閉 AMP。
    2. inputs:
        - cfg (dict|None): 設定檔；可含 train.amp_dtype ∈ {'none','fp16','bf16','auto'}。
    3. return:
        - dtype (torch.dtype|None): torch.bfloat16 / torch.float16 / None。
    """
    choice = None
    if cfg is not None and "train" in cfg and "amp_dtype" in cfg["train"]:
        choice = str(cfg["train"]["amp_dtype"]).lower()
    if choice == "none":
        return None
    if choice == "fp16":
        return torch.float16
    if choice == "bf16":
        return torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None

# =========================================================
# 類別先驗估計與動態 Threshold 搜尋
# =========================================================
@torch.no_grad()
def infer_class_prior(train_loader, num_classes: int, device: str):
    """
    1. 說明:
        以 train_loader 的標籤頻率估計類別先驗分佈（class prior），回傳總和為 1 的向量。
    2. inputs:
        - train_loader (DataLoader): 訓練資料載入器（回傳 y 可展平成 [B]）。
        - num_classes (int): 類別數。
        - device (str): 'cuda' 或 'cpu'（輸出張量放置位置）。
    3. return:
        - prior (FloatTensor): shape=[C] 的先驗分佈，元素皆 >0 且總和=1。
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, yb in train_loader:
        yb = yb.view(-1).cpu()
        binc = torch.bincount(yb, minlength=num_classes).to(torch.float64)
        counts[:num_classes] += binc[:num_classes]
    counts = counts.clamp(min=1)
    prior = (counts / counts.sum()).to(torch.float32).to(device)
    prior = prior.clamp_min(1e-8)
    prior = prior / prior.sum()
    return prior

#---------------------
# Optimizer（容錯 + fused 安全啟用）
# -----------------------------
def build_optimizer(model, cfg):
    """
    1. 說明:
        建立 AdamW 最佳化器；CUDA 上嘗試啟用 fused 版本（不支援則回退）。
    2. inputs:
        - model (nn.Module): 目標模型。
        - cfg (dict): 設定（train.lr, train.weight_decay）。
    3. return:
        - opt (torch.optim.Optimizer): 已建立的 AdamW。
    """
    lr = _as_float(cfg["train"].get("lr", 1e-3), "train.lr", default=1e-3)
    wd = _as_float(cfg["train"].get("weight_decay", 0.0), "train.weight_decay", default=0.0)
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=True)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

# -----------------------------
# Warmup 只在前 warmup_steps 生效；之後不再 step（避免覆蓋 CollapseGuard 的手動 LR 衰減）
# -----------------------------
def build_warmup_scheduler(optimizer, steps_per_epoch: int, cfg):
    """
    1. 說明:
        建立只在前 `warmup_epochs * steps_per_epoch` 生效的線性 warmup 調度器；
        warmup 結束後 `.step()` 不再改變 LR（避免覆蓋其他手動衰減策略）。
    2. inputs:
        - optimizer (Optimizer): 目標最佳化器。
        - steps_per_epoch (int): 每個 epoch 的 step 數。
        - cfg (dict): 設定（train.warmup_epochs）。
    3. return:
        - sched (object): 具有 `.step()` 的包裹調度器（過了 warmup 就 no-op）。
    """


    warmup_steps = int(cfg["train"].get("warmup_epochs", 0)) * max(1, steps_per_epoch)

    def lr_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        # 線性 warmup 到 1.0
        return min(1.0, step / max(1, warmup_steps))

    inner = LambdaLR(optimizer, lr_lambda)

    class _WarmupOnly:
        def __init__(self, opt, sched, warm_steps):
            self.opt = opt
            self.sched = sched
            self.warm_steps = warm_steps
            self._step = 0
        def step(self):
            # 只在 warmup 期間才真的 step；之後 no-op
            if self.warm_steps <= 0:
                return
            if self._step < self.warm_steps:
                self.sched.step()
            self._step += 1
        @property
        def last_epoch(self):
            return getattr(self.sched, "last_epoch", -1)

    return _WarmupOnly(optimizer, inner, warmup_steps)

# -----------------------------
# 類別權重（處理不平衡）
# -----------------------------
@torch.no_grad()
def infer_class_weights(train_loader, num_classes: int, device: str):
    """
    1. 說明:
        由訓練集中各類別頻率推得 class weights，並正規化使平均為 1。
    2. inputs:
        - train_loader (DataLoader): 訓練資料載入器（回傳 y 可展平成 [B]）。
        - num_classes (int): 類別數。
        - device (str): 權重張量要放置的裝置。
    3. return:
        - cw (FloatTensor): shape=[C] 的類別權重（平均值=1）。
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, yb in train_loader:
        yb = yb.view(-1).cpu()
        binc = torch.bincount(yb, minlength=num_classes).to(torch.float64)
        counts[:num_classes] += binc[:num_classes]
    counts = counts.clamp(min=1)
    total = counts.sum()
    cw = (total / counts).to(torch.float32).to(device)
    # 正規化到平均 1（避免 loss 尺度飄太大）
    cw = cw * (num_classes / cw.sum())
    return cw

def find_best_threshold_by_fbeta(
    y_true,
    y_score,
    *,
    beta: float = 0.5,
    grid_points: int = 201,
    task: str = "auto",
    reg_true_threshold: float = 0.0,
    average: str = "macro",
):
    """
    1. 說明:
        以單一函式同時支援「分類」與「回歸 → 二元化」的門檻搜尋（F-beta 最優）。
        - task='cls'：直接對 y_score 掃門檻；若 y_score∈[0,1] 用線性網格，否則用分位數網格。
        - task='reg'：先以 reg_true_threshold 將 y_true 二元化，再對連續 y_score 以分位數掃門檻。
        - task='auto'：若 y_true 僅含 0/1（或四捨五入後為 0/1）→ 視為分類；否則視為回歸。

    2. inputs:
        - y_true (array-like): 真值；分類時為 {0,1}，回歸時為連續值。
        - y_score (array-like): 分數/機率/連續預測，shape=[N]。
        - beta (float): F-beta 的 β，β<1 偏重 Precision（預設 0.5）。
        - grid_points (int): 門檻掃描點數（>=2）。分類且 y_score∈[0,1] 時為線性網格；其他情況用分位數網格。
        - task (str): 'auto' | 'cls' | 'reg'。
        - reg_true_threshold (float): 回歸模式下將 y_true 二元化的門檻；y_true >= 此值 → 1。
        - average (str): 傳給 sklearn.fbeta_score 的 average（預設 'macro'）。

    3. return:
        - best_thr (float): 最佳門檻。
        - info (dict): {"fbeta": 最佳Fbeta, "beta": beta, "task": 使用模式, "avg": average}
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    # 清理 NaN/Inf
    m = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[m], y_score[m]
    if y_true.size == 0:
        return 0.5, {"fbeta": 0.0, "beta": float(beta), "task": "empty", "avg": average}

    # 模式判定
    if task == "auto":
        uniq = np.unique(y_true)
        is_binary_like = np.all(np.isin(uniq, [0, 1])) or np.all(np.isin(np.unique(np.round(uniq)), [0, 1]))
        task_used = "cls" if is_binary_like else "reg"
    else:
        task_used = str(task).lower()

    # 構造二元真值
    if task_used == "reg":
        y_true_bin = (y_true >= float(reg_true_threshold)).astype(int)
    else:
        # 寬鬆接受 0/1 或 0.0/1.0；其餘數值以 0.5 二元化避免報錯
        if np.any((y_true != 0) & (y_true != 1)):
            y_true_bin = (y_true >= 0.5).astype(int)
        else:
            y_true_bin = y_true.astype(int)

    grid_points = max(2, int(grid_points))

    # 候選門檻：機率走 [0,1] 線性；否則走分位數（對 logits/連續分數較穩）
    if task_used == "cls" and np.all((y_score >= 0.0) & (y_score <= 1.0)):
        thr_list = np.linspace(0.0, 1.0, grid_points)
    else:
        qs = np.linspace(0.0, 1.0, grid_points)
        thr_list = np.quantile(y_score, qs)
    thr_list = np.unique(thr_list)

    # 掃描
    default_thr = 0.5 if task_used == "cls" else float(np.median(y_score))
    best_thr, best_val = default_thr, -1.0
    for t in thr_list:
        yhat = (y_score >= t).astype(int)
        val = fbeta_score(y_true_bin, yhat, beta=beta, average=average, zero_division=0)
        if val > best_val:
            best_val, best_thr = float(val), float(t)

    return float(best_thr), {"fbeta": float(best_val), "beta": float(beta), "task": task_used, "avg": average}


# -----------------------------
# AMP GradScaler（依 dtype）
# -----------------------------
def build_grad_scaler(dtype):
    return amp.GradScaler(enabled=(dtype == torch.float16))

# -----------------------------
# Trainer factory
# -----------------------------
def get_trainer(cfg):
    """
    1. 說明:
        依任務型態（classification/regression）回傳對應的 `train_one_fold` 函式。
    2. inputs:
        - cfg (dict): 設定檔（需能由 `get_task_type` 判斷任務）。
    3. return:
        - train_one_fold (Callable): 單一 fold 的訓練流程實作。
    """
    task = get_task_type(cfg)
    if task == "classification":
        from train.training.trainers.classification import train_one_fold
        return train_one_fold
    elif task == "regression":
        from train.training.trainers.regression import train_one_fold
        return train_one_fold
    else:
        raise ValueError(f"[trainer_factory] Unknown task type: {task}")


# =========================================================
# 溫度校準（Temperature Scaling
# =========================================================
import torch.nn.functional as F
def fit_temperature_ce(logits, y_true, max_iter=50):
    """
    1. 說明:
        溫度校準（Temperature Scaling, CE 版）。尋找標量溫度 T，使
        CE( logits / T, y_true ) 最小化。通常用於多分類 softmax 機率的後校準。
    2. inputs:
        - logits (Tensor): shape=[N,C] 的未經 softmax 之輸出分數（可為任意 dtype/裝置）。
        - y_true (Tensor): shape=[N] 的整數標籤。
        - max_iter (int): LBFGS 的最大迭代次數（預設 50）。
    3. return:
        - T (Tensor): 單一標量張量（與 logits 同裝置），可用於 `(logits / T)` 進行校準。
    """
    T = torch.nn.Parameter(torch.ones(1, device=logits.device, dtype=torch.float32))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    y_true = y_true.to(logits.device).long()

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy((logits.float()) / T.clamp_min(1e-4), y_true, reduction="mean")
        loss.backward()
        return loss

    opt.step(closure)
    return T.detach()
