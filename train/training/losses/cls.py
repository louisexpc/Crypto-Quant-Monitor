# train/training/losses/cls.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class BCEWithLogitsFocalLoss(nn.Module):
    """
    1. 說明:
        二分類 Focal Loss（以 BCEWithLogits 為基底）+ Confidence Penalty（負熵懲罰）。
        兼容模型輸出:
          - 單一二元 logit: [B] 或 [B,1]
          - 兩類 logits:    [B,2]  → 內部自動轉為單一二元 logit（logit(p1)=z1-z0）
        亦兼容 target:
          - 標籤索引: [B]、值∈{0,1}
          - 二元向量: [B,1] 或 one-hot [B,2]（自動取正類欄位）

    2. inputs:
        gamma (float): Focal 指數 γ（1~3；預設 2.0）—越大越聚焦難例。
        alpha (float|None): 正類權重 α（0~1；負類=1-α；None 不使用）。正類稀少可 0.6~0.8。
        conf_penalty (float): 負熵懲罰係數 λ（典型 0.02~0.05）。抑制過度自信、降低 collapse。
        reduction (str): 'mean' | 'sum' | 'none'（預設 'mean'）。
        normalized (bool): reduction='mean' 時用有效權重和做分母（梯度尺度更穩）。
        eps (float): 數值安全項（預設 1e-8）。

    3. return:
        loss (Tensor): 依 reduction 回傳 scalar 或逐樣本損失。
    """
    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[float] = None,
                 conf_penalty: float = 0.0,
                 reduction: str = "mean",
                 normalized: bool = True,
                 eps: float = 1e-8):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.conf_penalty = float(conf_penalty)
        self.reduction = reduction
        self.normalized = bool(normalized)
        self.eps = float(eps)

    def _to_binary_logit_and_target(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1. 說明:
            將各種形狀的 logits/target 正規化為:
              - logits_bin: [B] 單一二元 logit
              - y:          [B] ∈ {0,1} 的 float
        2. inputs:
            logits (Tensor): [B], [B,1], 或 [B,2]
            target (Tensor): [B], [B,1], 或 [B,2]（one-hot 亦可）
        3. return:
            (logits_bin, y)
        """
        # ---- logits → 單一二元 logit ----
        if logits.ndim == 2 and logits.size(1) == 2:
            # 將兩類 logits 映射為正類的 logit：logit(p1)=z1-z0
            logits_bin = logits[:, 1] - logits[:, 0]
        elif logits.ndim == 2 and logits.size(1) == 1:
            logits_bin = logits.squeeze(1)
        elif logits.ndim == 1:
            logits_bin = logits
        else:
            raise ValueError(f"[FocalLoss] Unexpected logits shape: {tuple(logits.shape)}; expected [B], [B,1], or [B,2].")

        # ---- target → [B] 0/1 float ----
        if target.ndim == 2:
            if target.size(1) == 2:
                # one-hot → 取正類欄位 (或 argmax 也可以，這裡支持軟 one-hot)
                y = target[:, 1]
            elif target.size(1) == 1:
                y = target.squeeze(1)
            else:
                raise ValueError(f"[FocalLoss] Unexpected target shape: {tuple(target.shape)}; expected [B], [B,1], or [B,2].")
        else:
            y = target
        # 允許 long/float；最終統一為 float 0/1
        if y.dtype.is_floating_point:
            y = y.clamp(0, 1).float()
        else:
            y = y.float()
        return logits_bin, y

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        1. 說明:
            計算二分類 Focal-BCE + 負熵懲罰。自動支援 2-logit 輸出與 one-hot 標籤。
        2. inputs:
            logits (Tensor): [B], [B,1] 或 [B,2]
            target (Tensor): [B], [B,1] 或 [B,2]（one-hot 亦可）
        3. return:
            loss (Tensor): 依 reduction 回傳
        """
        logits, y = self._to_binary_logit_and_target(logits, target)

        # 基底 BCE（逐樣本）
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")  # [B]

        # 機率與 focal 因子
        p = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)            # [B]
        p_t = torch.where(y > 0.5, p, 1.0 - p)                                # [B]
        focal = torch.pow(1.0 - p_t, self.gamma)                              # [B]

        # alpha_t
        if self.alpha is None:
            alpha_t = torch.ones_like(p_t)
        else:
            alpha_t = torch.where(y > 0.5,
                                  torch.full_like(p_t, self.alpha),
                                  torch.full_like(p_t, 1.0 - self.alpha))

        # 逐樣本 focal-BCE
        loss_i = alpha_t * focal * ce  # [B]

        # ---- Confidence Penalty： -H(p) ----
        if self.conf_penalty > 0.0:
            neg_entropy = p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)   # [B]
            if self.reduction == "none":
                loss_i = loss_i + self.conf_penalty * neg_entropy
            elif self.reduction == "sum":
                return loss_i.sum() + self.conf_penalty * neg_entropy.sum()
            else:  # mean
                if self.normalized:
                    denom = alpha_t.sum().clamp_min(self.eps)
                    focal_mean = loss_i.sum() / denom
                    cp_mean = (alpha_t * neg_entropy).sum() / denom
                    return focal_mean + self.conf_penalty * cp_mean
                else:
                    return loss_i.mean() + self.conf_penalty * neg_entropy.mean()

        # ---- 聚合 ----
        if self.reduction == "none":
            return loss_i
        if self.reduction == "sum":
            return loss_i.sum()
        if self.normalized:
            denom = alpha_t.sum().clamp_min(self.eps)
            return loss_i.sum() / denom
        else:
            return loss_i.mean()

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
    cfg_cls_loss = cfg["loss"]["cls"]
    use_focal = cfg_cls_loss["use_focal"]

    if use_focal:
        return BCEWithLogitsFocalLoss(gamma=float(cfg_cls_loss["focal_gamma"]),
                                      alpha=float(cfg_cls_loss["focal_alpha"]) or None,
                                      conf_penalty=float(cfg_cls_loss["conf_penalty"]),
                                      reduction=str(cfg_cls_loss["reduction"]),
                                      normalized= bool(cfg_cls_loss["normalized"])
                                    )
    else:
        return nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=cfg_cls_loss["label_smoothing"])
