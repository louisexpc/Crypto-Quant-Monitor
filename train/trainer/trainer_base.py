# trainer_base.py
import torch
import torch.nn.functional as F

from torch import amp
from sklearn.metrics import roc_curve

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_utils.init_train import setup_cuda_acceleration
setup_cuda_acceleration()

# -----------------------------
# 任務/裝置/AMP dtype 公用
# -----------------------------
def get_task_type(cfg: dict) -> str:
    """優先讀 cfg.task.type；否則看 num_classes 推斷。"""
    if "task" in cfg and "type" in cfg["task"]:
        return str(cfg["task"]["type"]).lower()
    if "target" in cfg and "type" in cfg["target"]:
        return str(cfg["target"]["type"]).lower()
    nc = int(cfg["model"].get("num_classes", 1))
    return "classification" if nc >= 2 else "regression"

def amp_dtype(cfg: dict | None = None):
    """
    由 YAML 控制 AMP dtype：
      train.amp_dtype ∈ {"none","fp16","bf16","auto"}
    - "none": 關閉 autocast
    - "fp16": torch.float16
    - "bf16": torch.bfloat16
    - "auto": 預設走 fp16；只有你明確指定 bf16 才會用 bf16
    """
    choice = None
    if cfg is not None and "train" in cfg and "amp_dtype" in cfg["train"]:
        choice = str(cfg["train"]["amp_dtype"]).lower()

    if choice == "none":
        return None
    if choice == "fp16":
        return torch.float16
    if choice == "bf16":
        # 僅當前裝置宣稱支援時才允許
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # 不支援則回退 fp16
        return torch.float16

    # "auto" 或未指定 → 保守使用 fp16
    if torch.cuda.is_available():
        return torch.float16
    return None  # 無 GPU 時直接關閉 AMP

def build_optimizer(model, cfg):
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=True)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def build_warmup_scheduler(optimizer, steps_per_epoch: int, cfg):
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = int(cfg["train"]["warmup_epochs"]) * max(1, steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            # 你也可以改成 linear decay / cosine
            return 1.0
    return LambdaLR(optimizer, lr_lambda)

# -----------------------------
# 分類公用：class weights / 門檻 / 溫度
# -----------------------------
@torch.no_grad()
def infer_class_weights(train_loader, num_classes: int, device: str):
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

def fit_temperature_ce(logits, y_true, max_iter=50):
    T = torch.nn.Parameter(torch.ones(1, device=logits.device))  # fp32
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)
    y_true = y_true.to(logits.device).long()

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy((logits.float()) / T, y_true, reduction="mean")  # ★ 強制 fp32
        loss.backward()
        return loss

    opt.step(closure)
    return T.detach()

# -----------------------------
# AMP GradScaler（依 dtype）
# -----------------------------
def build_grad_scaler(dtype):
    return amp.GradScaler(enabled=(dtype == torch.float16))




def get_trainer(cfg):
    task = get_task_type(cfg)
    if task == "classification":
        from .trainer_cls import train_one_fold
        return train_one_fold
    elif task == "regression":
        from .trainer_reg import train_one_fold
        return train_one_fold
    else:
        raise ValueError(f"[trainer_factory] Unknown task type: {task}")