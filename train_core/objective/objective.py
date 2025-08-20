# objective.py

from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
import optuna
import yaml
from copy import deepcopy

from .objective_utils import (get_task_type, suggest_rolling_and_cv, suggest_cat, suggest_float,
                             suggest_int, get_enabled_feature_names, suggest_model_hparams, make_folds)
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# --- Trainer / Data / Model ---
from trainer.trainer_base import get_trainer
from build_feature_loader.dataloader import make_loaders_for_fold
from models.model_factory import build_model
from utils.init_train import set_seed, setup_cuda_acceleration


# ======================================================================
# Trial 得分計算（依 primary_metric / direction）
# ======================================================================
def compute_trial_score(result: dict, cfg: dict) -> float:
    """
    回傳單一 fold 的分數（由 caller 負責做平均）。
    - 分類 primary_metric（常見）：threshold_macro_f05 / macro_f1 / acc（通常 direction='maximize'）
    - 回歸 primary_metric：mixed（minimize）、pearson（maximize）、val_loss（minimize）
    """
    primary = str(cfg["objective"].get("primary_metric", "threshold_macro_f05")).lower()
    direction = str(cfg["objective"].get("direction", "maximize")).lower()
    task_type = get_task_type(cfg)

    if task_type == "classification":
        if primary == "threshold_macro_f05":
            # 需要兩類；trainer 已在二分類時寫入 threshold_metrics
            score = result.get("threshold_metrics", {}).get("f_05_macro", None)
            if score is None:
                # 沒有 threshold 版就 fallback macro_f1
                score = result.get("test_metrics", {}).get("test_macro_f1", 0.0)
        elif primary == "macro_f1":
            score = result.get("test_metrics", {}).get("test_macro_f1", 0.0)
        elif primary in ["acc", "accuracy"]:
            score = result.get("test_metrics", {}).get("test_acc", 0.0)
        else:
            raise ValueError(f"[objective] Unsupported primary_metric for classification: {primary}")

        return float(score) if direction == "maximize" else float(-score)

    else:
        # regression
        if primary == "mixed":
            # α·EMA-MSE + β·(1−Pearson)（越小越好）
            score = result.get("best_val_mixed", None)
            if score is None:
                # 後備：用測試 rmse
                score = result.get("test_metrics_reg", {}).get("rmse", np.inf)
            # direction 建議為 minimize；若外部誤設 maximize，這裡仍回傳「帶符號」供 study 使用
            return float(score) if direction == "minimize" else float(-score)

        elif primary == "pearson":
            score = result.get("best_val_pearson", None)
            if score is None:
                score = result.get("test_metrics_reg", {}).get("pearson", 0.0)
            return float(score) if direction == "maximize" else float(-score)

        elif primary in ["val_loss", "loss"]:
            score = result.get("test_metrics_reg", {}).get("test_loss", np.inf)
            return float(score) if direction == "minimize" else float(-score)

        else:
            raise ValueError(f"[objective] Unsupported primary_metric for regression: {primary}")


# ======================================================================
# 主目標函式（給 Optuna）
# ======================================================================
def objective(trial: optuna.Trial, base_cfg: dict, df, run_dir: Path, pt_bundle=None) -> float: 
    setup_cuda_acceleration()
    cfg = copy.deepcopy(base_cfg)
    task_type = get_task_type(cfg)
    target_col = "label" if task_type == "classification" else "target"

    trial_dir = run_dir / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cfg = suggest_rolling_and_cv(trial, cfg)
    cfg.setdefault("data", {})
    cfg["data"]["train_val_split"] = float(cfg["cv"]["train_val_split"])

    device = "cuda" if (cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu"
    set_seed(int(cfg.get("seed", 42)) + trial.number)

    # ==== 訓練超參 ====
    cfg["train"]["lr"] = suggest_float(trial, "lr", cfg["train"]["lr"])
    cfg["train"]["weight_decay"] = suggest_float(trial, "weight_decay", cfg["train"]["weight_decay"])
    cfg["train"]["batch_size"] = int(cfg["train"]["batch_size"])  # 固定值即可
    cfg["train"]["grad_clip"] = suggest_float(trial, "grad_clip", cfg["train"]["grad_clip"])

    # ==== 分類專用超參 ====
    if task_type == "classification":
        thr = cfg["train"]["threshold"]
        if isinstance(thr, list) and len(thr) == 2:
            cfg["train"]["threshold"] = trial.suggest_float("threshold", float(thr[0]), float(thr[1]))
        else:
            cfg["train"]["threshold"] = float(thr)

    # ==== 回歸專用超參 ====
    if task_type == "regression":
        for key in ["alpha", "ema_decay", "beta"]:
            val = cfg["loss"][key]
            if isinstance(val, list) and len(val) == 2:
                cfg["loss"][key] = trial.suggest_float(f"loss.{key}", float(val[0]), float(val[1]))
            else:
                cfg["loss"][key] = float(val)
        # beta auto fallback
        if "beta" not in cfg["loss"] or cfg["loss"]["beta"] is None:
            cfg["loss"]["beta"] = max(0.0, 1.0 - cfg["loss"]["alpha"])

    # ==== 模型超參 ====
    cfg = suggest_model_hparams(trial, cfg)

    # ==== 特徵選擇 ====
    feat_pool = get_enabled_feature_names(cfg, df.columns)
    n_features = len(feat_pool)

    # ==== folds ====
    folds = make_folds(df, cfg)

    # ==== trainer ====
    train_one_fold = get_trainer(cfg)

    # ==== 每 fold 訓練 ====
    fold_scores = []
    for i, fold in enumerate(folds):
        fold_dir = trial_dir / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        # tr_loader, va_loader, te_loader, _ = make_loaders_for_fold(
        #     df, feat_pool, target_col, fold, cfg
        # )

        tr_loader, va_loader, te_loader, info = make_loaders_for_fold(
            df, feat_pool, target_col, fold, cfg, also_XGB=cfg["also_XGB"]
        )
        # ★ fold 專屬 cfg，避免污染其他 fold
        cfg_fold = deepcopy(cfg)

        # ★ 這裡一定要用「同一個大小寫」的 key：XGB
        assert "XGB" in info and info["XGB"] is not None, \
            "[objective] info['XGB'] 不存在；請確認 make_loaders_for_fold(..., also_XGB=True)"
        cfg_fold["_xgb_pack"] = info["XGB"]

        model = build_model(cfg, n_features)


        # _, result = train_one_fold(model, tr_loader, va_loader, te_loader, cfg, device, i, fold_dir)
        _, result = train_one_fold(
            model, tr_loader, va_loader, te_loader,
            cfg_fold, device=device, fold_id=i, export_dir=fold_dir
        )

        score = compute_trial_score(result, cfg)
        fold_scores.append(float(score))
        trial.report(float(score), step=i)

        # 若要啟用 Optuna pruning
        if cfg["objective"]["enable_prune"]:
            if trial.should_prune():
                raise optuna.TrialPruned()

    # ==== 紀錄與儲存 ====
    mean_score = float(np.mean(fold_scores))
    trial.set_user_attr("selected_features", feat_pool)
    trial.set_user_attr("n_features", n_features)
    trial.set_user_attr("fold_scores", fold_scores)
    trial.set_user_attr("task_type", task_type)
    trial.set_user_attr("primary_metric", str(cfg["objective"]["primary_metric"]))
    trial.set_user_attr("direction", str(cfg["objective"]["direction"]))
    trial.set_user_attr("target_col", target_col)

    trial_cfg_path = trial_dir / f"trial_config_{cfg['objective']['primary_metric']}={mean_score:.6g}.yaml"
    with open(trial_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    return mean_score