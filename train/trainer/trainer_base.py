# trainer_base.py
import torch
import torch.nn.functional as F
from torch import amp
from sklearn.metrics import roc_curve
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_utils.init_train import setup_cuda_acceleration
setup_cuda_acceleration()

# -----------------------------
# 小工具：把可能是 list/tuple 的數值轉成純 float
# -----------------------------
def _as_float(x, name="value", default=None):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        # 這裡你也可以選擇取中位數或最大/最小，依你的搜尋策略
        return float(x[0])
    if default is not None:
        return float(default)
    raise TypeError(f"[trainer_base] {name} expects a float or list/tuple, got {type(x)}")

# -----------------------------
# 任務/裝置/AMP dtype 公用
# -----------------------------
def get_task_type(cfg: dict) -> str:
    if "task" in cfg and "type" in cfg["task"]:
        return str(cfg["task"]["type"]).lower()
    if "target" in cfg and "type" in cfg["target"]:
        return str(cfg["target"]["type"]).lower()
    nc = int(cfg["model"].get("num_classes", 1))
    return "classification" if nc >= 2 else "regression"

def amp_dtype(cfg: dict | None = None):
    choice = None
    if cfg is not None and "train" in cfg and "amp_dtype" in cfg["train"]:
        choice = str(cfg["train"]["amp_dtype"]).lower()
    if choice == "none":
        return None
    if choice == "fp16":
        return torch.float16
    if choice == "bf16":
        return torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    # auto：偏向 bf16（若支援），否則 fp16；比原本更聰明一點
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None

# -----------------------------
# Optimizer（容錯 + fused 安全啟用）
# -----------------------------
def build_optimizer(model, cfg):
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
    from torch.optim.lr_scheduler import LambdaLR

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
# 分類公用：class weights / 門檻 / 溫度
# -----------------------------
@torch.no_grad()
def infer_class_weights(train_loader, num_classes: int, device: str):
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

def find_best_threshold_by_auc(y_true, y_prob_pos):
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
    # logits 可為 [N,C] 的任意 dtype；這裡統一在 fp32 上做
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

# -----------------------------
# AMP GradScaler（依 dtype）
# -----------------------------
def build_grad_scaler(dtype):
    return amp.GradScaler(enabled=(dtype == torch.float16))

# -----------------------------
# Trainer factory
# -----------------------------
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
