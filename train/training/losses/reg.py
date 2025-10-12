# train/training/losses/reg.py
import torch
import torch.nn as nn

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

def build_regression_loss(cfg):
    cfg_reg_loss = cfg["loss"]["reg"]
    name = cfg_reg_loss["name"]
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        delta = float(cfg_reg_loss["huber_delta"])
        return nn.HuberLoss(delta=delta)
    # default: emamse_pearson
    a = float(cfg_reg_loss["alpha"])
    b = float(cfg_reg_loss["beta"])
    d = float(cfg_reg_loss["ema_decay"])
    eps = float(1e-8)
    return EMAMSE_PearsonLoss(alpha=a, beta=b, ema_decay=d, eps=eps)



