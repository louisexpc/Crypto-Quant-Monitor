"""
trainer_cls.py
--------------
分類任務專用的訓練迴圈（single fold）。支援：
1) CE / Focal-BCE 損失（含 class weight、label smoothing、Confidence Penalty）
2) 類別分佈對齊（distribution alignment）
3) 自動溫度校準（temperature scaling, CE 版）
4) 二分類動態門檻（threshold）搜尋：AUC-Youden 或 F-beta
5) 早停（可選用 val_loss 或 macro F0.5 作為主指標）
6) 訓練/驗證/測試的完整指標與可視化匯出
7) CollapseGuard：PPR/熵 監控與自救（λ_cp 調整、LR 衰減、可選回滾最佳權重）

注意：
- 檔案假設在 CUDA/GPU 環境下執行（會 assert）
- 時序資料的資料載入器需保證 time-aware 的切分策略（外部保證）
"""
from __future__ import annotations
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from sklearn.metrics import fbeta_score
from typing import Optional, Tuple, Dict, Callable

from .trainer_base import (
    amp_dtype, build_optimizer, build_warmup_scheduler, build_grad_scaler,
    infer_class_weights, fit_temperature_ce, find_best_threshold_by_auc
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 匯出/視覺化/度量
from train_utils.compute_export_metrices import save_fold_metrics, plot_test_eval
from train_utils.metrics_cls import compute_cls_metrics  # 統一計算 acc/f1/f0.5/mcc 等

# =========================================================
# Loss：BCEWithLogits + Focal + Confidence Penalty（支援 [B], [B,1], [B,2]）
# =========================================================
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

# =========================================================
# 高速批次迭代（GPU 預載資料集直切）
# =========================================================
def _iter_batches(loader, device: str, batch_size: int):
    """
    若 DataLoader 的 Dataset 已將整個資料預載為 GPU/CUDA 張量（例如 EventDataset/SeqDataset），
    則直接使用連續切片方式產生 batch，避免 DataLoader 逐樣本索引與堆疊帶來的小 kernel/CPU 開銷。
    否則回退為原始 loader 迭代。
    """
    try:
        ds = getattr(loader, "dataset", None)
        if ds is not None and hasattr(ds, "X") and hasattr(ds, "y"):
            X, y = ds.X, ds.y
            if isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):
                # 僅當 X/y 已在目標裝置上時走快路徑
                if X.device.type == device and y.device.type == device:
                    n = int(y.shape[0])
                    bs = max(1, int(batch_size))
                    for s in range(0, n, bs):
                        e = min(n, s + bs)
                        yield X[s:e], y[s:e]
                    return
    except Exception:
        pass  # 任意例外都回退到原 loader
    # 回退：沿用原本 DataLoader 逐批產生
    for xb, yb in loader:
        yield xb, yb

# =========================================================
# CollapseGuard：PPR/平均熵 監控與自救
# =========================================================
def prob_entropy_from_logits_binary(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    1. 說明:
        將二分類 logits 轉為正類機率 p1，並計算每筆樣本的二元熵 H(p1)。
        支援 logits 形狀: [B], [B,1], [B,2]（若為 [B,2] 視為二元 softmax）。
    2. inputs:
        - logits: torch.Tensor
    3. return:
        - p1: torch.Tensor, 形狀 [B]，每筆正類機率
        - H:  torch.Tensor, 形狀 [B]，每筆二元熵
    """
    if logits.ndim == 1:
        z = logits
        p1 = torch.sigmoid(z)
    elif logits.ndim == 2 and logits.shape[-1] == 1:
        z = logits[..., 0]
        p1 = torch.sigmoid(z)
    elif logits.ndim == 2 and logits.shape[-1] == 2:
        p = F.softmax(logits, dim=-1)
        p1 = p[:, 1]
    else:
        raise ValueError(f"Unsupported logits shape for binary task: {tuple(logits.shape)}")

    p1 = p1.clamp_(1e-6, 1 - 1e-6)
    H = -(p1 * torch.log(p1) + (1 - p1) * torch.log(1 - p1))
    return p1, H


def init_final_bias_to_prior(module: nn.Module, pos_rate: float) -> None:
    """
    1. 說明:
        以資料先驗正類率 π 初始化最後一層 bias，使初期平均機率接近 π，減少 PPR=0 的假警報。
    2. inputs:
        - module:  具有 .bias 的最後輸出層 (nn.Linear out=1 或 nn.Conv1d out_channels=1)
        - pos_rate: 訓練集正類率 π (0<π<1)
    3. return:
        - None
    """
    pi = float(max(1e-4, min(1 - 1e-4, pos_rate)))
    b0 = math.log(pi / (1 - pi))
    with torch.no_grad():
        if getattr(module, "bias", None) is not None:
            module.bias.fill_(b0)


class CollapseGuard:
    """
    1. 說明:
        訓練時監控「預測為正類的比例 PPR」與「平均熵」，在異常時提供自救手段：
        - 提高 λ_cp（若 loss 模組有 conf_penalty）
        - 衰減學習率（不低於 min_lr）
        - 觸發回呼（可回滾最佳權重）
        同時支援：
        * 動態門檻（每 epoch 後用驗證的 best_val_thresh 回寫）
        * 暖身期、熵門檻、觸發冷卻，避免早期與重複干擾
        * EMA 平滑 PPR 與熵

    2. inputs:
        pos_threshold (float):     PPR 計算門檻（可由 set_pos_threshold 動態更新）
        ppr_warn_band (tuple):     告警帶 (low, high)，超出連續 warn_patience 步→warn
        warn_patience (int):       告警耐心
        ppr_extreme_band (tuple):  極端帶 (low, high)，超出連續 extreme_patience 步→觸發
        extreme_patience (int):    觸發耐心
        cp_boost_factor (float):   觸發時 λ_cp 乘法因子
        lr_decay (float):          觸發時學習率乘法因子
        max_conf_penalty (float):  λ_cp 上限
        min_lr (float):            學習率不低於此值
        on_trigger (Callable):     觸發回呼（可回滾最佳權重）
        smoothing (float):         EMA 係數（對前值權重，建議 0.9~0.99；0=不用 EMA）
        warmup_steps (int):        暖身步數（暖身內不告警/觸發）
        entropy_hi (float):        只在「高熵」時才視為值得告警（二分類最大 ~0.693；建議 0.60）
        cooldown_steps (int):      觸發後的冷卻步數，避免連環觸發
        verbose (bool):            是否列印 warn 與觸發訊息

    3. return:
        透過 on_batch_end()/on_epoch_end() 回傳監控 dict
    """
    def __init__(self,
                 pos_threshold: float = 0.5,
                 ppr_warn_band: Tuple[float, float] = (0.05, 0.60),
                 warn_patience: int = 50,
                 ppr_extreme_band: Tuple[float, float] = (0.02, 0.98),
                 extreme_patience: int = 100,
                 cp_boost_factor: float = 1.25,
                 lr_decay: float = 0.5,
                 max_conf_penalty: float = 0.20,
                 min_lr: float = 1e-6,
                 on_trigger: Optional[Callable[[Dict], None]] = None,
                 smoothing: float = 0.95,
                 warmup_steps: int = 200,
                 entropy_hi: float = 0.60,
                 cooldown_steps: int = 200,
                 verbose: bool = False) -> None:
        # 門檻與帶寬
        self.pos_threshold = float(pos_threshold)
        self.ppr_low, self.ppr_high = map(float, ppr_warn_band)
        self.ext_low, self.ext_high = map(float, ppr_extreme_band)
        self.warn_patience = int(warn_patience)
        self.extreme_patience = int(extreme_patience)

        # 自救配置
        self.cp_boost_factor = float(cp_boost_factor)
        self.lr_decay = float(lr_decay)
        self.max_conf_penalty = float(max_conf_penalty)
        self.min_lr = float(min_lr)
        self.on_trigger = on_trigger

        # 平滑/門檻/冷卻
        self.alpha = float(smoothing)           # EMA 對前值的權重
        self.warmup_steps = int(warmup_steps)
        self.entropy_hi = float(entropy_hi)
        self.cooldown_steps = int(cooldown_steps)
        self.verbose = bool(verbose)

        # 狀態
        self._warn_streak = 0
        self._extreme_streak = 0
        self._ema_ppr: Optional[float] = None
        self._ema_entropy: Optional[float] = None
        self._step = 0
        self._last_trigger_step = -10**9

    def _ema(self, prev: Optional[float], value: float) -> float:
        """
        1. 說明:
            指標的 EMA 更新。
        2. inputs:
            - prev:  上一個 EMA 值或 None
            - value: 本次原始值
        3. return:
            - new_ema: 更新後 EMA 值
        """
        if self.alpha <= 0 or prev is None:
            return value
        return self.alpha * prev + (1 - self.alpha) * value

    @torch.no_grad()
    def on_batch_end(self,
                     logits: torch.Tensor,
                     loss_module: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     model: Optional[nn.Module] = None) -> Dict:
        """
        1. 說明:
            每個 batch 結束呼叫，更新 PPR/熵 的 (E)MA，必要時觸發自救。
        2. inputs:
            - logits: 本 batch 模型輸出（[B], [B,1] 或 [B,2]）
            - loss_module: 若含 .conf_penalty，觸發時會調整
            - optimizer:   觸發時衰減 LR（不低於 min_lr）
            - model:       供回呼使用（如回滾最佳權重）
        3. return:
            - info: 指標與觸發資訊 dict
        """
        self._step += 1

        # 機率與熵
        p1, H = prob_entropy_from_logits_binary(logits)
        ppr = float((p1 >= self.pos_threshold).float().mean().item())
        ent = float(H.mean().item())

        # EMA
        self._ema_ppr = self._ema(self._ema_ppr, ppr)
        self._ema_entropy = self._ema(self._ema_entropy, ent)
        ppr_use = self._ema_ppr if self.alpha > 0 else ppr
        ent_use = self._ema_entropy if self.alpha > 0 else ent

        # Gate: 暖身 / 高熵 / 冷卻
        ready = (self._step > self.warmup_steps)
        high_entropy = (ent_use >= self.entropy_hi)
        in_cooldown = (self._step - self._last_trigger_step) < self.cooldown_steps

        # Warn 與 Extreme 判斷
        outside = ready and high_entropy and ((ppr_use < self.ppr_low) or (ppr_use > self.ppr_high))
        self._warn_streak = self._warn_streak + 1 if outside else 0
        warn_hit = (self._warn_streak >= self.warn_patience)

        extreme = ready and high_entropy and ((ppr_use < self.ext_low) or (ppr_use > self.ext_high))
        self._extreme_streak = self._extreme_streak + 1 if (extreme and not in_cooldown) else 0
        trigger = (self._extreme_streak >= self.extreme_patience) and not in_cooldown

        did_adjust_cp = False
        did_decay_lr = False

        # Optional: 列印 warn
        if self.verbose and outside:
            print(f"[WARN step={self._step}] PPR_ema={ppr_use:.3f} Entropy_ema={ent_use:.3f}")

        if trigger:
            # 1) 提高 λ_cp
            if hasattr(loss_module, "conf_penalty"):
                old_cp = float(loss_module.conf_penalty)
                new_cp = min(self.max_conf_penalty, (old_cp * self.cp_boost_factor) if old_cp > 0 else 0.02)
                if new_cp != old_cp:
                    loss_module.conf_penalty = new_cp
                    did_adjust_cp = True

            # 2) 衰減 LR（不低於 min_lr）
            for pg in optimizer.param_groups:
                if "lr" in pg and pg["lr"] > 0:
                    new_lr = max(self.min_lr, float(pg["lr"]) * self.lr_decay)
                    if new_lr < pg["lr"]:
                        pg["lr"] = new_lr
                        did_decay_lr = True

            # 3) 回呼
            if self.on_trigger is not None:
                ctx = dict(step=self._step,
                           ppr=ppr, ppr_ema=self._ema_ppr,
                           entropy=ent, entropy_ema=self._ema_entropy,
                           adjusted_cp=did_adjust_cp, decayed_lr=did_decay_lr,
                           loss_module=loss_module, optimizer=optimizer, model=model)
                try:
                    self.on_trigger(ctx)
                except Exception as e:
                    print(f"[CollapseGuard] on_trigger error: {e}")

            # 重置計數 + 記錄冷卻
            self._extreme_streak = 0
            self._warn_streak = 0
            self._last_trigger_step = self._step

            if self.verbose:
                print(f"[TRIGGER step={self._step}] ppr_ema={ppr_use:.3f} ent_ema={ent_use:.3f} | "
                      f"cp_adj={did_adjust_cp} lr_dec={did_decay_lr}")

        return {
            "step": self._step,
            "ppr": ppr, "ppr_ema": self._ema_ppr,
            "entropy": ent, "entropy_ema": self._ema_entropy,
            "warn": bool(warn_hit), "extreme": bool(extreme),
            "triggered": bool(trigger),
            "did_adjust_cp": did_adjust_cp, "did_decay_lr": did_decay_lr,
        }

    @torch.no_grad()
    def on_epoch_end(self) -> Dict:
        """
        1. 說明:
            每個 epoch 收尾呼叫，回傳當前 EMA 指標與連續計數摘要。
        2. inputs:
            - None
    3. return:
            - info: dict
        """
        return {
            "ppr_ema": self._ema_ppr,
            "entropy_ema": self._ema_entropy,
            "warn_streak": self._warn_streak,
            "extreme_streak": self._extreme_streak,
        }

    def set_pos_threshold(self, thr: float) -> None:
        """
        1. 說明:
            讓 Guard 能在每次驗證後，改用當輪驗證得到的最佳 threshold（動態門檻）。
        2. inputs:
            - thr: float, 新的門檻
        3. return:
            - None
        """
        self.pos_threshold = float(thr)


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
        - 訓練：支援 AMP、warmup、梯度裁剪、分佈對齊、Confidence Penalty。
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
        raise ValueError(f"[trainer_cls] 模型 out_dim={out_dim} != num_classes={num_class}。請調整 head。")

    # class weights
    class_weights = infer_class_weights(train_loader, num_class, device)
    loss_fn = build_classification_loss(cfg, class_weights=class_weights)

    # ---- 額外正則項（分佈對齊 / CE 的信心懲罰）----
    dist_align_weight = float(cfg["train"].get("dist_align_weight", 0.0))
    conf_penalty_w    = float(cfg["train"].get("confidence_penalty", 0.0))
    prior_mode        = str(cfg["train"].get("dist_prior", "train")).lower()
    if dist_align_weight > 0.0:
        if prior_mode == "uniform":
            class_prior = torch.full((num_class,), 1.0 / max(1, num_class), dtype=torch.float32, device=device)
        else:
            class_prior = infer_class_prior(train_loader, num_class, device)
    else:
        class_prior = None

    # ---- CollapseGuard 設定（只在二分類時啟用）----
    guard_cfg = cfg.get("guard", cfg.get("collapse_guard", {})) or {}
    guard_enabled = bool(guard_cfg.get("enabled", True)) and (num_class == 2)
    guard = None
    if guard_enabled:
        # PPR 門檻建議與實際決策門檻一致；若使用固定門檻可讀取 train.threshold
        thr_mode = str(cfg["train"].get("threshold_mode", "auto_auc")).lower()
        if thr_mode == "fixed" and (cfg["train"].get("threshold") is not None):
            pos_thr = float(cfg["train"]["threshold"])
        else:
            pos_thr = float(guard_cfg.get("pos_threshold", 0.5))

        # 可選回滾：在觸發時載回當前最佳權重
        best_state_holder = {"state": None}
        def _on_trigger(ctx: Dict):
            if bool(guard_cfg.get("restore_on_trigger", True)) and best_state_holder["state"] is not None:
                try:
                    ctx["model"].load_state_dict(best_state_holder["state"])
                    print("[CollapseGuard] restored best weights.")
                except Exception as e:
                    print(f"[CollapseGuard] restore error: {e}")

        guard = CollapseGuard(
            pos_threshold=pos_thr,
            ppr_warn_band=tuple(guard_cfg.get("ppr_warn_band", (0.05, 0.60))),
            warn_patience=int(guard_cfg.get("warn_patience", 50)),
            ppr_extreme_band=tuple(guard_cfg.get("ppr_extreme_band", (0.02, 0.98))),
            extreme_patience=int(guard_cfg.get("extreme_patience", 100)),
            cp_boost_factor=float(guard_cfg.get("cp_boost_factor", 1.25)),
            lr_decay=float(guard_cfg.get("lr_decay", 0.5)),
            max_conf_penalty=float(guard_cfg.get("max_conf_penalty", 0.20)),
            min_lr=float(guard_cfg.get("min_lr", 1e-6)),
            on_trigger=_on_trigger,
            smoothing=float(guard_cfg.get("smoothing", 0.95)),
            warmup_steps=int(guard_cfg.get("warmup_steps", steps_per_epoch * 2)),
            entropy_hi=float(guard_cfg.get("entropy_hi", 0.60)),
            cooldown_steps=int(guard_cfg.get("cooldown_steps", max(steps_per_epoch, 200))),
            verbose=bool(guard_cfg.get("verbose", False)),
)
    else:
        best_state_holder = {"state": None}

    # ---- Early-stop 狀態 ----
    best_epoch = 0
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    best_state_holder["state"] = best_state  # 讓 CollapseGuard 能回滾
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
        last_guard_info = None

        for Xb, yb in _iter_batches(train_loader, device, int(cfg["train"]["batch_size"])):
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()

            if not printed_shape:
                print("[trainer_cls] DEBUG batch X shape =", tuple(Xb.shape))
                printed_shape = True

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", dtype=dtype, enabled=(dtype is not None)):
                logits = model(Xb)
                loss = loss_fn(logits, yb)

                # (1) 分佈對齊：KL( mean(p) || prior )  — 僅多類/二類 softmax 的情況
                if dist_align_weight > 0.0 and class_prior is not None and logits.shape[-1] > 1:
                    probs = torch.softmax(logits, dim=1).float()  # [B, C]
                    p_mean = probs.mean(dim=0)                     # [C]
                    kl = F.kl_div((p_mean + 1e-8).log(), class_prior, reduction="batchmean")
                    loss = loss + dist_align_weight * kl

                # (2) CE 路線的信心懲罰：期望的負熵（logits.shape[-1] > 1）
                #    使用 BCEWithLogitsFocalLoss 時，CP 已內嵌於 loss；此處不重複。
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

            # CollapseGuard 監控與自救（訓練後、作用於後續 step）
            if guard is not None:
                info = guard.on_batch_end(logits.detach(), loss_fn, optimizer, model=model)
                last_guard_info = info
                if info["warn"]:
                    print(f"[WARN step={info['step']}] PPR_ema={info['ppr_ema']:.3f} Entropy_ema={info['entropy_ema']:.3f}")
                if info["triggered"]:
                    print(f"[TRIGGER] λ_cp→{getattr(loss_fn,'conf_penalty',0.0):.4f}; LR decayed.")

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
            elif thr_mode == "fixed":
                cfg_thresh = cfg["train"].get("threshold", None)
                curr_val_thresh = float(cfg_thresh) if cfg_thresh is not None else 0.5
            else:
                curr_val_thresh, _ = find_best_threshold_by_auc(y_va, y_score_va)
            yhat_va = (y_score_va >= curr_val_thresh).astype(int)
        else:
            yhat_va = probs_all.argmax(axis=1)
            curr_val_thresh = None

        m_va = compute_cls_metrics(y_va, yhat_va)
        if guard is not None and curr_val_thresh is not None:
            guard.set_pos_threshold(float(curr_val_thresh))

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
            best_state_holder["state"] = best_state  # 讓 CollapseGuard 能回滾最新最佳
            best_T = T_epoch.detach().clone()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"{prefix}Early stop at epoch {epoch} | best_epoch={best_epoch} | val_f1={best_val_f1:.4f} | val_f05={best_val_f_05:.4f}")
                break

        # ---- CollapseGuard epoch 摘要 ----
        guard_epoch_info = guard.on_epoch_end() if guard is not None else {}
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
            # CollapseGuard logs
            "guard_ppr_ema": guard_epoch_info.get("ppr_ema", np.nan),
            "guard_entropy_ema": guard_epoch_info.get("entropy_ema", np.nan),
            "guard_warn_streak": guard_epoch_info.get("warn_streak", np.nan),
            "guard_extreme_streak": guard_epoch_info.get("extreme_streak", np.nan),
            "guard_last_ppr": last_guard_info.get("ppr", np.nan) if last_guard_info else np.nan,
            "guard_last_entropy": last_guard_info.get("entropy", np.nan) if last_guard_info else np.nan,
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

    with torch.no_grad(), amp.autocast(device_type="cuda", dtype=dtype, enabled=(dtype is not None)):
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

    # ---- 組合回傳 ----
    result = {
        "history": history,
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
    save_fold_metrics(history, save_dir=export_dir, prefix=f"fold_{fold_id}_")
    plot_test_eval(
        y_true=y_te, y_pred=yhat_te, y_prob=y_prob_te,
        class_names=cfg["model"].get("class_names", None),
        save_dir=export_dir,
        prefix=f"fold_{fold_id}_",
        threshold=float(best_thresh) if (is_binary_test and best_thresh is not None) else None
    )
    return model, result
