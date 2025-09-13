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
from .objective_utils import (get_task_type, suggest_rolling_and_cv, suggest_cat, suggest_float,
                             suggest_int, suggest_model_hparams, make_folds,
                             _format_score_tag, _safe_rename_trial_dir)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# --- Trainer / Data / Model ---
from trainer.trainer_base import get_trainer
from build_feature_loader.dataloader import make_time_loaders_for_fold, make_event_loaders_for_fold
from models.model_factory import build_model
from train_utils.init_train import set_seed, setup_cuda_acceleration
from train_utils.compute_export_metrices import save_cv_summary


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
    if result is None:
        return -1e9  # 或 0.0，依你的 direction
    
    if task_type == "classification":
        if primary == "threshold_macro_f05":
            # 需要兩類；trainer 已在二分類時寫入 threshold_metrics
            score = result.get("threshold_metrics", {}).get("f_05_macro", None)
            if score is None:
                # 沒有 threshold 版就 fallback macro_f1
                score = result.get("test_metrics", {}).get("test_macro_f1", 0.0)
        elif primary == "macro_f1":
            score = result.get("test_metrics", {}).get("test_macro_f1", 0.0)
        
        elif primary == "mcc":
            score = result.get("test_metrics", {}).get("test_mcc", 0.0)

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

    # 以 worker_tag 區分輸出資料夾與追蹤資訊（不調整 batch_size）
    worker_tag = os.environ.get("WORKER_TAG", "").strip()   # 來自 main_train.run_multi()
    orig_bs = int(cfg["train"]["batch_size"])
    # 可選：印一下方便追蹤
    if worker_tag:
        print(f"[objective] worker={worker_tag} | batch_size={cfg['train']['batch_size']} (orig={orig_bs})")

    # === NEW: 把 worker_tag / batch_size 寫進 trial 屬性，便於事後追蹤 ===
    trial.set_user_attr("worker_tag", worker_tag or "single")
    trial.set_user_attr("batch_size", int(cfg["train"]["batch_size"]))

    # === NEW: 將不同 worker 的輸出分到不同子資料夾（你前面已加，留著提醒）===
    if worker_tag:
        trial_dir = run_dir / worker_tag / f"trial_{trial.number:03d}"
    else:
        trial_dir = run_dir / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cfg = suggest_rolling_and_cv(trial, cfg)
    cfg.setdefault("data", {})
    if "cv" in cfg and "train_val_split" in cfg["cv"]:
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
        thr_mode = str(cfg.get("train", {}).get("threshold_mode", "auto_auc")).lower()
        thr = cfg.get("train", {}).get("threshold", None)
        if thr_mode == "fixed":
            # 只有 fixed 模式才對 threshold 做搜尋/強制數值
            if isinstance(thr, list) and len(thr) == 2:
                cfg["train"]["threshold"] = trial.suggest_float("threshold", float(thr[0]), float(thr[1]))
            elif thr is None:
                cfg["train"]["threshold"] = 0.5  # 預設固定門檻
            else:
                cfg["train"]["threshold"] = float(thr)
        else:
            # 動態掃描（auto_auc/auto_fbeta）不需要固定 threshold
            cfg["train"]["threshold"] = None

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
    
    # ==== 分數差分（FFD）超參 ====
    if str(cfg["label"]["ret_type"]).lower() == "fractionally":
        cfg["label"]["fracdiff"]["d"] = suggest_float(
            trial,
            "fracdiff.d",
            cfg["label"]["fracdiff"]["d"]
        )
        # 方便日後追蹤：把本次 d 寫進 trial 屬性
        trial.set_user_attr("fracdiff_d", float(cfg["label"]["fracdiff"]["d"]))

    # ==== 特徵選擇 ====
    # 交由 dataloader 在 runtime 決定與計算；此處不預先決定 n_features
    label_mode = str(cfg.get("label", {}).get("mode", "")).lower()
    n_features = None

    # ==== 事件模式：一次計算全量 15m feature grid，供各 fold 重用 ====
    pre_feat_df = None  # dataloader 會在需要時計算

    # ==== folds ====
    folds = make_folds(df, cfg)

    # ==== trainer ====
    train_one_fold = get_trainer(cfg)

    # ==== 每 fold 訓練 ====
    fold_scores, fold_results = [], []
    fold_models_for_infer = []  # (model, fold_dict, result) for post-infer export
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

        # Update target_col from dataloader info (may be 'y_cls'/'y_reg')
        target_col = info.get("target_col", target_col)

        feature_columns = info.get("feat_cols")
        if n_features is None:
            n_features = len(feature_columns)
        model = build_model(cfg, n_features, feature_columns)

        model_trained, result = train_one_fold(
            model, tr_loader, va_loader, te_loader,
            cfg_fold, device=device, fold_id=i, export_dir=fold_dir
        )
        fold_results.append(result)
        # 保留本 fold 最佳權重模型、fold 定義與結果（溫度/門檻等）供事後推論用
        try:
            fold_models_for_infer.append((model_trained, fold, result))
        except Exception:
            pass
        score = compute_trial_score(result, cfg)
        fold_scores.append(float(score))
        trial.report(float(score), step=i)

        # 若要啟用 Optuna pruning
        if cfg["objective"]["enable_prune"]:
            if trial.should_prune():

                raise optuna.TrialPruned()

    # ==== 這裡印出 avg 的 metrics（VAL / TEST）====
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
    else:  # regression
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
            
    save_cv_summary(fold_results, export_dir=trial_dir, task_type=task_type)  # ★ 產出 cv_summary.json

    # === 依任務型別決定是否改名 ===
    if task_type == "classification":
        # 你的 trainer_cls.py 回傳鍵是 "test_mcc"；保留 "mcc" 後備
        mcc_cv = test_avg.get("test_mcc", test_avg.get("mcc", np.nan))
        trial.set_user_attr("test_mcc_avg", float(mcc_cv) if np.isfinite(mcc_cv) else None)
        if np.isfinite(mcc_cv):
            tag = _format_score_tag("mcc", mcc_cv, digits=3, signed=True)  # 例：__mcc+0.123
            trial_dir = _safe_rename_trial_dir(trial_dir, [tag])

    elif task_type == "regression":
        pearson_cv = test_avg.get("pearson", np.nan)
        trial.set_user_attr("test_pearson_avg", float(pearson_cv) if np.isfinite(pearson_cv) else None)
        if np.isfinite(pearson_cv):
            tag = _format_score_tag("pearson", pearson_cv, digits=4, signed=True)
            trial_dir = _safe_rename_trial_dir(trial_dir, [tag])



    # ==== 紀錄與儲存 ====
    mean_score = float(np.mean(fold_scores))
    trial.set_user_attr("selected_features", feature_columns if 'feature_columns' in locals() else None)
    trial.set_user_attr("n_features", int(n_features) if n_features is not None else None)
    trial.set_user_attr("fold_scores", fold_scores)
    trial.set_user_attr("task_type", task_type)
    trial.set_user_attr("primary_metric", str(cfg["objective"]["primary_metric"]))
    trial.set_user_attr("direction", str(cfg["objective"]["direction"]))
    trial.set_user_attr("target_col", target_col)

    trial_cfg_path = trial_dir / f"trial_config_{cfg['objective']['primary_metric']}={mean_score:.6g}.yaml"
    with open(trial_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    # ===== Optional: 對指定期間的全部 TBM 事件做推論，並把預測欄位寫回 TBM CSV =====
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
            # 若 csv_path_override 為 null/空字串，回退到 cfg.label.tbm_csv_path
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
                    threshold_override=(post_infer.get("threshold_override") if isinstance(post_infer.get("threshold_override", None), (int, float)) else None),
                )
                print(f"[PostInfer] Saved TBM with predictions: {save_csv}")
                debug_lines.append("status=ok")
            except Exception as e2:
                debug_lines.append(f"status=error")
                debug_lines.append(f"error={type(e2).__name__}: {e2}")
                # 將錯誤記錄到 trial 目錄，方便排查
                try:
                    with open(trial_dir / "post_infer_error.txt", "w", encoding="utf-8") as ef:
                        ef.write("\n".join(debug_lines))
                except Exception:
                    pass
        # 不論是否執行，寫入 debug 紀錄
        try:
            with open(trial_dir / "post_infer_debug.txt", "w", encoding="utf-8") as dfp:
                dfp.write("\n".join(debug_lines))
        except Exception:
            pass
    except Exception as e:
        print(f"[PostInfer][WARN] skip due to error: {e}")

    return mean_score
