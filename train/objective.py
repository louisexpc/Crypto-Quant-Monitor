# objective.py

import copy
import numpy as np
import torch
import optuna
from pathlib import Path

from trainer import train_one_fold
from utils.dataloader import make_loaders_for_fold, make_loaders_for_fold_from_pt
from models.model_factory import build_model
from utils.anchored_loader import make_anchored_monthly_folds, make_rolling_monthly_folds
from hyp_utils import set_seed, build_feature_pool, sample_k_subset


def suggest_float(trial, name, val, log_names=frozenset(("lr", "weight_decay"))):
    if isinstance(val, list) and len(val) == 2:
        low, high = float(val[0]), float(val[1])
        use_log = (name in log_names) and (low > 0.0 and high > 0.0)
        return trial.suggest_float(name, low, high, log=use_log)
    return float(val)

def suggest_int(trial, name, val):
    if isinstance(val, list) and len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
        return int(trial.suggest_int(name, int(val[0]), int(val[1])))
    return int(val)

def suggest_cat(trial, name, vals):
    return trial.suggest_categorical(name, tuple(vals))


def folds_type(df, cfg):
    cv_type = cfg["cv"]["type"]
    if cv_type == "OddEven":
        from utils.dataloader import make_two_month_folds
        return make_two_month_folds(df.index, cfg["cv"]["start_month"] + "-01")
    elif cv_type == "Anchored":
        return make_anchored_monthly_folds(df.index, cfg["cv"]["start_month"] + "-01",
                                           cfg["cv"].get("embargo_hours", 24),
                                           min_train_days=cfg["cv"].get("min_train_days", 30))
    elif cv_type == "Rolling":
        return make_rolling_monthly_folds(df.index, cfg["cv"]["train_months"], cfg["cv"]["embargo_hours"])
    else:
        raise ValueError(f"Unknown fold type: {cv_type}")


def objective(trial: optuna.Trial, base_cfg: dict, df, run_dir: Path, pt_bundle=None) -> float:
    trial_dir = run_dir / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(base_cfg)
    device = "cuda" if (cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu"
    set_seed(int(cfg.get("seed", 42)) + trial.number)

    # ----- 抽超參數 -----
    cfg["train"]["lr"] = suggest_float(trial, "lr", cfg["train"]["lr"])
    cfg["train"]["weight_decay"] = suggest_float(trial, "weight_decay", cfg["train"]["weight_decay"])
    cfg["train"]["epochs"] = suggest_int(trial, "epochs", cfg["train"]["epochs"])

    grad_clip = cfg["train"]["grad_clip"]
    cfg["train"]["grad_clip"] = trial.suggest_float("grad_clip", *grad_clip)

    cfg["train"]["batch_size"] = cfg["train"]["batch_size"]

    cfg["sequence"]["seq_len"] = suggest_cat(trial, "seq_len", cfg["sequence"]["seq_len"])

    # if isinstance(cfg["label"]["flat_band_bps"], list):
    #     cfg["label"]["flat_band_bps"] = suggest_int(trial, "flat_band_bps", cfg["label"]["flat_band_bps"])
    
    # model 架構參數
    for param in ["hidden_size", "n_layers"]:
        val = cfg["model"][param]
        if isinstance(val, list):
            if len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
                cfg["model"][param] = trial.suggest_int(param, int(val[0]), int(val[1]))
            else:
                cfg["model"][param] = suggest_cat(trial, param, val)
        else:
            cfg["model"][param] = int(val)
    
    dp = cfg["model"].get("dropout", 0.0)
    if isinstance(dp, list) and len(dp) == 2:
        cfg["model"]["dropout"] = trial.suggest_float("dropout", dp[0], dp[1])
    else:
        cfg["model"]["dropout"] = float(dp)

    cfg["model"]["bidirectional"] = False  # 固定單向

    # ----- 特徵抽樣 -----
    feat_pool = build_feature_pool(df.columns.to_list(), cfg)
    always_on = cfg.get("features", {}).get("always_on", [])
    k_low, k_high = cfg.get("features", {}).get("selection", {}).get("k_range", [128, 256])
    selected_feats = sample_k_subset(trial, feat_pool, always_on, (k_low, k_high))
    n_features = len(selected_feats)

    folds = folds_type(df, cfg)

    make_loader_fn = make_loaders_for_fold_from_pt if cfg["data"].get("use_pt_cache", False) else make_loaders_for_fold



    
    fold_scores = []
    for i, fold in enumerate(folds):
        tr_loader, va_loader, te_loader, _ = make_loader_fn(df, selected_feats, "label", fold, cfg)
        model = build_model(cfg, n_features)

        fold_dir = trial_dir / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        _, result = train_one_fold(
            model=model,
            train_loader=tr_loader,
            val_loader=va_loader,
            test_loader=te_loader,
            cfg=cfg,
            device=device,
            fold_id=i,
            export_dir=fold_dir, 
        )

        # f1 = float(result.get("best_val_macro_f1", 0.0))
        # fold_scores.append(f1)
        # trial.report(f1, step=i)
        # # if trial.should_prune():
        # #     raise optuna.TrialPruned()

        prec = float(result.get("best_val_prec", 0.0))
        fold_scores.append(prec)
        trial.report(prec, step=i)

    # mean_f1 = float(np.mean(fold_scores)) if fold_scores else 0.0
    mean_prec = float(np.mean(fold_scores)) 



    trial.set_user_attr("selected_features", selected_feats)
    trial.set_user_attr("n_features", n_features)
    trial.set_user_attr("fold_scores", fold_scores)

    import yaml
    trial_cfg_path = trial_dir / f"trial_config_prec_{mean_prec}.yaml"
    with open(trial_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    return mean_prec
