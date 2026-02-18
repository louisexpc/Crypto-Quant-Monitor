# train/training/losses/cls.py
from __future__ import annotations

import torch.nn as nn


def build_classification_loss(cfg, class_weights):
    """Build classification loss function.

    Args:
        cfg: Global config dict.
        class_weights: Tensor of class weights or None.

    Returns:
        Configured `nn.Module` classification loss.
    """
    cfg_cls_loss = (cfg.get("loss", {}) or {}).get("cls", {}) or {}
    use_class_weight = bool(cfg_cls_loss.get("use_class_weight", True))
    weight = class_weights if use_class_weight else None
    return nn.CrossEntropyLoss(weight=weight)
