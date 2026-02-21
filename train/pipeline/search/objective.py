"""Optuna objective wrapper that delegates to the pipeline trial runner."""

from __future__ import annotations

import copy
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict

import optuna
import numpy as np
import torch

from train.pipeline.search.space import (
    get_task_type,
    suggest_sequence_and_cv,
    suggest_model_hparams,
    suggest_float,
    make_folds,
)
from train.pipeline.trial_runner import run_trial

__all__ = ["objective"]


def _set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for one trial run.

    Args:
        seed: Random seed value.
        deterministic: Whether to enable deterministic CUDA behavior.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        pass
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated.*",
        category=UserWarning,
        module="pandas_ta",
    )


def _prepare_config(trial: optuna.Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg = suggest_sequence_and_cv(trial, cfg)
    cfg.setdefault("data", {})
    if "cv" in cfg and "train_val_split" in cfg["cv"]:
        cfg["data"]["train_val_split"] = float(cfg["cv"]["train_val_split"])
    cfg = suggest_model_hparams(trial, cfg)

    # 標準訓練超參（lr / weight_decay / grad_clip）
    train_cfg = cfg.get("train", {})
    if "lr" in train_cfg:
        train_cfg["lr"] = suggest_float(trial, "lr", train_cfg["lr"])
    if "weight_decay" in train_cfg:
        train_cfg["weight_decay"] = suggest_float(trial, "weight_decay", train_cfg["weight_decay"])
    if "grad_clip" in train_cfg:
        train_cfg["grad_clip"] = suggest_float(trial, "grad_clip", train_cfg["grad_clip"])
    if "batch_size" in train_cfg:
        train_cfg["batch_size"] = int(train_cfg["batch_size"])

    return cfg


def _apply_task_specific_adjustments(trial: optuna.Trial, cfg: Dict[str, Any]) -> None:
    task_type = get_task_type(cfg)
    if task_type == "classification":
        thr_mode = str(cfg.get("train", {}).get("threshold_mode", "auto_auc")).lower()
        thr = cfg.get("train", {}).get("threshold", None)
        if thr_mode == "fixed":
            if isinstance(thr, list) and len(thr) == 2:
                cfg["train"]["threshold"] = trial.suggest_float("threshold", float(thr[0]), float(thr[1]))
            elif thr is None:
                cfg["train"]["threshold"] = 0.5
            else:
                cfg["train"]["threshold"] = float(thr)
        else:
            cfg["train"]["threshold"] = None
    else:
        if "loss" in cfg:
            for key in ["alpha", "ema_decay", "beta"]:
                val = cfg["loss"].get(key)
                if isinstance(val, list) and len(val) == 2:
                    cfg["loss"][key] = trial.suggest_float(f"loss.{key}", float(val[0]), float(val[1]))
                elif val is not None:
                    cfg["loss"][key] = float(val)
            if "beta" not in cfg["loss"] or cfg["loss"]["beta"] is None:
                cfg["loss"]["beta"] = max(0.0, 1.0 - cfg["loss"].get("alpha", 0.0))

    if str(cfg.get("label", {}).get("ret_type", "")).lower() == "fractionally":
        frac = cfg["label"].get("fracdiff", {})
        cfg["label"].setdefault("fracdiff", frac)
        cfg["label"]["fracdiff"]["d"] = trial.suggest_float(
            "fracdiff.d",
            float(frac.get("d", 0.3)),
            float(frac.get("d", 0.3)),
        )
        trial.set_user_attr("fracdiff_d", float(cfg["label"]["fracdiff"]["d"]))


def objective(trial: optuna.Trial, base_cfg: Dict[str, Any], df, run_dir: Path) -> float:
    cfg = _prepare_config(trial, base_cfg)
    _apply_task_specific_adjustments(trial, cfg)

    worker_tag = os.environ.get("WORKER_TAG", "").strip()
    trial.set_user_attr("worker_tag", worker_tag or "single")
    trial.set_user_attr("batch_size", int(cfg["train"]["batch_size"]))

    if worker_tag:
        trial_dir = run_dir / worker_tag / f"trial_{trial.number:03d}"
    else:
        trial_dir = run_dir / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(cfg.get("seed", 42))
    effective_seed = base_seed + trial.number
    _set_seed(effective_seed, deterministic=bool(cfg.get("deterministic", True)))

    folds = make_folds(df, cfg)

    result = run_trial(
        optuna_trial=trial,
        cfg=cfg,
        df=df,
        trial_dir=trial_dir,
        folds=folds,
        effective_seed=effective_seed,
    )

    return result.mean_score
