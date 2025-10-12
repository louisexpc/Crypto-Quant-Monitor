# train/training/hooks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Callable
import math
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


# def init_final_bias_to_prior(module: nn.Module, pos_rate: float) -> None:
#     """
#     1. 說明:
#         以資料先驗正類率 π 初始化最後一層 bias，使初期平均機率接近 π，減少 PPR=0 的假警報。
#     2. inputs:
#         - module:  具有 .bias 的最後輸出層 (nn.Linear out=1 或 nn.Conv1d out_channels=1)
#         - pos_rate: 訓練集正類率 π (0<π<1)
#     3. return:
#         - None
#     """
#     pi = float(max(1e-4, min(1 - 1e-4, pos_rate)))
#     b0 = math.log(pi / (1 - pi))
#     with torch.no_grad():
#         if getattr(module, "bias", None) is not None:
#             module.bias.fill_(b0)


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
# 溫度校準（Temperature Scaling
# =========================================================
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

