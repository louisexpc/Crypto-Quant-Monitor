# objective.py
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
import optuna
import yaml
from copy import deepcopy
import pandas as pd
import os
from .export_tbm_pred import export_tbm_predictions_for_trial
from .objective_utils import (
    get_task_type, suggest_rolling_and_cv, suggest_cat, suggest_float,
    suggest_int, suggest_model_hparams, make_folds,
    _format_score_tag, _safe_rename_trial_dir, save_trial_config,
    compute_trial_score
)
import sys
# --- Trainer / Data / Model ---
from train.trainer.trainer_base import get_trainer
from train.data.dataloaders.time_loader import make_time_loaders_for_fold
from train.data.dataloaders.event_loader import make_event_loaders_for_fold
from train.models.model_factory import build_model
from train.train_utils.init_train import set_seed, setup_cuda_acceleration
from train.train_utils.compute_export_metrices import save_cv_summary






def objective(trial: optuna.Trial, base_cfg: dict, df, run_dir: Path, pt_bundle=None) -> float:
    setup_cuda_acceleration()
    cfg = copy.deepcopy(base_cfg)
    task_type = get_task_type(cfg)
    target_col = "label" if task_type == "classification" else "target"

    # worker_tag（多進程時分開資料夾）
    worker_tag = os.environ.get("WORKER_TAG", "").strip()
    # orig_bs = int(cfg["train"]["batch_size"])
    # if worker_tag:
    #     print(f"[objective] worker={worker_tag} | batch_size={cfg['train']['batch_size']} (orig={orig_bs})")
    trial.set_user_attr("worker_tag", worker_tag or "single")
    trial.set_user_attr("batch_size", int(cfg["train"]["batch_size"]))

    # trial 目錄
    if worker_tag:
        trial_dir = run_dir / worker_tag / f"trial_{trial.number:03d}"
    else:
        trial_dir = run_dir / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # CV / rolling 可能受 trial 控制
    cfg = suggest_rolling_and_cv(trial, cfg)
    cfg.setdefault("data", {})
    if "cv" in cfg and "train_val_split" in cfg["cv"]:
        cfg["data"]["train_val_split"] = float(cfg["cv"]["train_val_split"])

    # 以「實際使用的 seed」做訓練；為了重現，在儲存 config 前也會覆寫 cfg["seed"]
    base_seed = int(cfg.get("seed", 42))
    effective_seed = base_seed + trial.number
    set_seed(effective_seed)

    device = "cuda" if (cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu"

    # ==== 訓練超參 ====
    cfg["train"]["lr"] = suggest_float(trial, "lr", cfg["train"]["lr"])
    cfg["train"]["weight_decay"] = suggest_float(trial, "weight_decay", cfg["train"]["weight_decay"])
    cfg["train"]["batch_size"] = int(cfg["train"]["batch_size"])  # 不動態調整
    cfg["train"]["grad_clip"] = suggest_float(trial, "grad_clip", cfg["train"]["grad_clip"])

    # ==== 分類專用 ====
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

    # ==== 回歸專用 ====
    if task_type == "regression":
        for key in ["alpha", "ema_decay", "beta"]:
            val = cfg["loss"][key]
            if isinstance(val, list) and len(val) == 2:
                cfg["loss"][key] = trial.suggest_float(f"loss.{key}", float(val[0]), float(val[1]))
            else:
                cfg["loss"][key] = float(val)
        if "beta" not in cfg["loss"] or cfg["loss"]["beta"] is None:
            cfg["loss"]["beta"] = max(0.0, 1.0 - cfg["loss"]["alpha"])

    # ==== 模型超參 ====
    cfg = suggest_model_hparams(trial, cfg)

    # ==== 分數差分（FFD） ====
    if str(cfg.get("label", {}).get("ret_type", "")).lower() == "fractionally":
        cfg["label"]["fracdiff"]["d"] = suggest_float(trial, "fracdiff.d", cfg["label"]["fracdiff"]["d"])
        trial.set_user_attr("fracdiff_d", float(cfg["label"]["fracdiff"]["d"]))

    # ==== 事件模式：一次算好供各 fold 用 ====
    pre_feat_df = None

    # ==== folds ====
    folds = make_folds(df, cfg)

    # ==== trainer ====
    train_one_fold = get_trainer(cfg)

    # ==== 每 fold 訓練 ====
    fold_scores, fold_results = [], []
    fold_models_for_infer = []  # (model, fold_dict, result)
    for i, fold in enumerate(folds):
        fold_dir = trial_dir / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        label_mode = str(cfg.get("label", {}).get("mode", "")).lower()
        if label_mode == "event_tbm":
            tr_loader, va_loader, te_loader, info = make_event_loaders_for_fold(
                df, [], fold, cfg, also_XGB=cfg["also_XGB"], pre_feat_df=pre_feat_df
            )
        else:
            tr_loader, va_loader, te_loader, info = make_time_loaders_for_fold(
                df, None, None, fold, cfg, also_XGB=cfg["also_XGB"], pre_feat_df=pre_feat_df
            )

        cfg_fold = deepcopy(cfg)
        assert "XGB" in info and info["XGB"] is not None, \
            "[objective] info['XGB'] 不存在；請確認 make_*_loaders_for_fold(..., also_XGB=True)"
        cfg_fold["_xgb_pack"] = info["XGB"]

        target_col = info.get("target_col", target_col)
        feature_columns = info.get("feat_cols")
        n_features = len(feature_columns) if feature_columns is not None else None

        model = build_model(cfg, n_features, feature_columns)

        model_trained, result = train_one_fold(
            model, tr_loader, va_loader, te_loader,
            cfg_fold, device=device, fold_id=i, export_dir=fold_dir
        )
        fold_results.append(result)
        try:
            # 將模型搬回 CPU，避免同時保留多個 fold 模型佔滿顯存
            if model_trained is not None:
                try:
                    model_trained = model_trained.to("cpu")
                finally:
                    try:
                        # 使用模組層的 torch，勿在函式內重新 import 以避免 Python 將 torch 視為區域變數
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
            fold_models_for_infer.append((model_trained, fold, result))
        except Exception:
            pass

        score = compute_trial_score(result, cfg)
        fold_scores.append(float(score))
        trial.report(float(score), step=i)

        if cfg["objective"]["enable_prune"] and trial.should_prune():
            raise optuna.TrialPruned()

    # ==== CV 平均 ====
    def _numeric_only(d):
        out = {}
        for k, v in (d or {}).items():
            if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                out[k] = float(v)
        return out

    def _avg(rows):
        pool = {}
        for d in rows:
            for k, v in _numeric_only(d).items():
                pool.setdefault(k, []).append(v)
        return {k: float(np.mean(vs)) for k, vs in pool.items()}

    if task_type == "classification":
        val_rows  = [r.get("val_metrics",  {}) for r in fold_results]
        test_rows = [r.get("test_metrics", {}) for r in fold_results]
    else:
        val_rows  = [r.get("val_metrics_reg",  {}) for r in fold_results]
        test_rows = [r.get("test_metrics_reg", {}) for r in fold_results]

    val_avg  = _avg(val_rows)
    test_avg = _avg(test_rows)

    print(f"\n[CV] {task_type.upper()} | folds={len(fold_results)}")
    if val_avg:
        print("[CV] VAL  avg:")
        for k in sorted(val_avg):
            print(f"  {k}: {val_avg[k]:.6g}")
    if test_avg:
        print("[CV] TEST avg:")
        for k in sorted(test_avg):
            print(f"  {k}: {test_avg[k]:.6g}")

    save_cv_summary(fold_results, export_dir=trial_dir, task_type=task_type)

    # === 依型別決定資料夾後綴改名 ===
    if task_type == "classification":
        mcc_cv = test_avg.get("test_mcc", test_avg.get("mcc", np.nan))
        trial.set_user_attr("test_mcc_avg", float(mcc_cv) if np.isfinite(mcc_cv) else None)
        if np.isfinite(mcc_cv):
            tag = _format_score_tag("mcc", mcc_cv, digits=3, signed=True)
            trial_dir = _safe_rename_trial_dir(trial_dir, [tag])

    elif task_type == "regression":
        pearson_cv = test_avg.get("pearson", np.nan)
        trial.set_user_attr("test_pearson_avg", float(pearson_cv) if np.isfinite(pearson_cv) else None)
        if np.isfinite(pearson_cv):
            tag = _format_score_tag("pearson", pearson_cv, digits=4, signed=True)
            trial_dir = _safe_rename_trial_dir(trial_dir, [tag])

    # ==== 只存「可重現」的那一份，命名沿用原本規則 ====
    mean_score = float(np.mean(fold_scores))

    # ★ 關鍵：把 cfg["seed"] 覆寫成「實際使用的 seed」，確保重跑能復現
    cfg["seed"] = int(effective_seed)

    trial_cfg_path = trial_dir / f"trial_config_{cfg['objective']['primary_metric']}={mean_score:.6g}.yaml"
    with open(trial_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    print(f"[objective] saved reproducible config -> {trial_cfg_path}")

    # ===== Optional: 事後 TBM 匯出（保持原邏輯）=====
    try:
        post_infer = (cfg.get("post_infer", {}) or {}).get("tbm_concat", {}) or {}
        debug_lines = []
        debug_lines.append(f"task_type={task_type}")
        debug_lines.append(f"post_infer_enabled={bool(post_infer.get('enabled', False))}")
        debug_lines.append(f"trial_dir={trial_dir}")
        debug_lines.append(f"n_fold_models={len(fold_models_for_infer)}")
        if bool(post_infer.get("enabled", False)) and task_type == "classification":
            ds, de = str(post_infer.get("date_start", "2023-01-01")), str(post_infer.get("date_end", "2025-08-01"))
            out_col = str(post_infer.get("output_column", "pred"))
            out_csv = str(post_infer.get("csv_path_override") or cfg.get("label", {}).get("tbm_csv_path"))
            s_tag = ds.replace('-', '')
            e_tag = de.replace('-', '')
            save_csv = trial_dir / f"tbm_with_{out_col}_{s_tag}_{e_tag}.csv"
            debug_lines.append(f"date_range=[{ds},{de}]")
            debug_lines.append(f"tbm_src={out_csv}")
            debug_lines.append(f"save_to={save_csv}")
            try:
                export_tbm_predictions_for_trial(
                    cfg=cfg,
                    df_index=pd.DatetimeIndex(df.index),
                    folds=folds,
                    fold_models=fold_models_for_infer,
                    date_start=ds,
                    date_end=de,
                    src_tbm_csv_path=out_csv,
                    save_to_path=str(save_csv),
                    output_column=out_col,
                    threshold_override=(
                        post_infer.get("threshold_override")
                        if isinstance(post_infer.get("threshold_override", None), (int, float))
                        else None
                    ),
                    decision_mode="both"
                )
                print(f"[PostInfer] Saved TBM with predictions: {save_csv}")
                debug_lines.append("status=ok")
            except Exception as e2:
                debug_lines.append("status=error")
                debug_lines.append(f"error={type(e2).__name__}: {e2}")
                try:
                    with open(trial_dir / "post_infer_error.txt", "w", encoding="utf-8") as ef:
                        ef.write("\n".join(debug_lines))
                except Exception:
                    pass
        try:
            with open(trial_dir / "post_infer_debug.txt", "w", encoding="utf-8") as dfp:
                dfp.write("\n".join(debug_lines))
        except Exception:
            pass
    except Exception as e:
        print(f"[PostInfer][WARN] skip due to error: {e}")

    return mean_score
