import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# === 回歸損失與工具 ===
class EMAMSE_PearsonLoss(nn.Module):
    """
    total_loss = alpha * EMA(MSE) + beta * (1 - PearsonCorr)
    注意：EMA 權重是沿著 batch 維度遞減，請盡量用時間順序的 Sampler。

    EMA_MSE: 關注數值接近      * α 
    Pearson: 關注趨勢方向接近  * β
    """
    def __init__(self, alpha=0.7, beta=0.3, ema_decay=0.9, eps=1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)

    def forward(self, y_pred, y_true):
        # y_pred, y_true: [B] or [B,1]
        y_pred = y_pred.squeeze(-1)
        y_true = y_true.squeeze(-1)

        # --- EMA-MSE ---
        err2 = (y_pred - y_true) ** 2  # [B]
        B = err2.numel()
        # w_t = (1 - d) * d^{B-1-t}; 最新樣本權重最大
        idx = torch.arange(B, device=err2.device, dtype=err2.dtype)
        w = (1.0 - self.ema_decay) * (self.ema_decay ** (B - 1 - idx))
        w = w / (w.sum() + self.eps)
        ema_mse = (err2 * w).sum()

        # --- PearsonCorr ---
        x = y_pred - y_pred.mean()
        y = y_true - y_true.mean()
        denom = (x.pow(2).sum().sqrt() * y.pow(2).sum().sqrt()).clamp_min(self.eps)
        corr = (x * y).sum() / denom
        # 對極端/常數情況做健壯化
        corr = corr.clamp(min=-1.0, max=1.0)

        return self.alpha * ema_mse + self.beta * (1.0 - corr)



# def pearson_corr_np(y_true, y_pred, eps=1e-12):
#     y_true = np.asarray(y_true).ravel()
#     y_pred = np.asarray(y_pred).ravel()
#     sy = y_true.std()
#     sp = y_pred.std()
#     if not np.isfinite(sy) or not np.isfinite(sp) or sy < eps or sp < eps:
#         return 0.0  # 或者回傳 np.nan 再由上層統一處理
#     return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)  # ★
    y_pred = np.asarray(y_pred, dtype=np.float64)  # ★
    mse  = float(np.mean((y_true - y_pred) ** 2))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    r = np.corrcoef(y_true, y_pred)[0, 1]
    if not np.isfinite(r):
        r = 0.0
    return {"mse": mse, "mae": mae, "rmse": rmse, "pearson": r}


def build_regression_loss(cfg):
    name = (cfg.get("loss", {}) or {}).get("name", "emamse_pearson").lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        delta = float(cfg["loss"].get("huber_delta", 1.0))
        return nn.HuberLoss(delta=delta)
    # default: emamse_pearson
    a = float(cfg["loss"].get("alpha", 0.7))
    b = float(cfg["loss"].get("beta", 0.3))
    d = float(cfg["loss"].get("ema_decay", 0.9))
    eps = float(cfg["loss"].get("pearson_eps", 1e-8))
    return EMAMSE_PearsonLoss(alpha=a, beta=b, ema_decay=d, eps=eps)


def get_task_type(cfg):
    # 兼容不同配置鍵
    if "task" in cfg and "type" in cfg["task"]:
        return cfg["task"]["type"].lower()
    if "target" in cfg and "type" in cfg["target"]:
        return cfg["target"]["type"].lower()
    # fallback：照舊用 num_classes 判斷
    nc = int(cfg["model"].get("num_classes", 1))
    return "classification" if nc >= 2 else "regression"


# === 回歸 → 分類（選配） ===
from sklearn.metrics import fbeta_score

def binarize_regression(y, threshold=0.0):
    # y >= threshold → 1； 否則 0
    y = np.asarray(y).reshape(-1)
    return (y >= float(threshold)).astype(int)

def find_best_threshold_for_regression(y_true_reg, y_pred_reg, fbeta=0.5, grid_points=101, true_threshold=0.0):
    """
    將回歸的 y 轉成 0/1（以 true_threshold），並對 y_pred 門檻掃描，選 F_beta 最高的門檻。
    回傳: best_thresh, best_fbeta
    """
    y_true_bin = binarize_regression(y_true_reg, threshold=true_threshold)
    # 掃描 y_pred 的分位數作為候選門檻（亦可改成線性區間）
    qs = np.linspace(0, 1, int(grid_points))
    cand = np.quantile(y_pred_reg, qs)
    best_t, best_f = 0.0, -1.0
    for t in cand:
        yhat = (y_pred_reg >= t).astype(int)
        f = fbeta_score(y_true_bin, yhat, beta=fbeta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f
