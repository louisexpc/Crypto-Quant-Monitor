# objective.py

import copy
import numpy as np
import torch
import optuna
from pathlib import Path

from trainer import train_one_fold
from utils.dataloader import FoldGenerator, make_loaders_for_fold

from models.model_factory import build_model
# from utils.anchored_loader import make_anchored_monthly_folds, make_rolling_monthly_folds, make_loaders_for_fold
from hyp_utils import set_seed, build_feature_pool, sample_k_subset

from utils.cuda_utils import setup_cuda_acceleration

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


def suggest_rolling_and_cv(trial, cfg):
    # 1) seq_len 先抽，因 embargo 下界依賴它
    seq_len_cfg = cfg["sequence"]["seq_len"]
    if isinstance(seq_len_cfg, list):
        seq_len = trial.suggest_categorical("seq_len", [int(x) for x in seq_len_cfg])
    else:
        seq_len = int(seq_len_cfg)
    cfg["sequence"]["seq_len"] = seq_len

    # 2) stride
    stride_space = [1, 2, 3, 6, 12]
    stride = trial.suggest_categorical("stride", stride_space)
    cfg["sequence"]["step"] = stride

    # 3) Rolling 訓練窗
    train_months = suggest_cat(trial, "cv.train_months", cfg["cv"]["train_months"])
    cfg["cv"]["train_months"] = int(train_months)

    # 4) 測試頻率
    test_freq = suggest_cat(trial, "cv.test_freq", cfg["cv"]["test_freq"])
    cfg["cv"]["test_freq"] = test_freq

    # 5) embargo（動態下界）
    bar_size_hours = 1  # 你的資料是 1h bar
    embargo_min = max(0, (seq_len - 1) * bar_size_hours)
    embargo_max_cfg = int(cfg["cv"]["embargo_hours"][1])
    embargo = trial.suggest_int("cv.embargo_hours", embargo_min, embargo_max_cfg, step=6)
    cfg["cv"]["embargo_hours"] = int(embargo)

    # 6) train/val split（附安全檢查）
    split_low, split_high = cfg["cv"]["train_val_split"]
    tv_split = trial.suggest_float("cv.train_val_split", float(split_low), float(split_high))
    cfg["cv"]["train_val_split"] = float(tv_split)

    return cfg

def folds_type(df, cfg):    
    cv_type = cfg["cv"]["type"]
    fold_g = FoldGenerator(dt_index=df.index, mode=cv_type,start_month=cfg["cv"]["start_month"])

    if cv_type == "OddEven":
        return fold_g.make_two_month_folds()
    
    elif cv_type == "Anchored":
        return fold_g.make_anchored_folds(
            embargo_hours=cfg["cv"].get("embargo_hours", 24),
            min_train_days=cfg["cv"].get("min_train_days", 30),
            test_freq=cfg["cv"].get("test_freq", "M")     # 加這個
        )

    elif cv_type == "Rolling":
        return fold_g.make_rolling_folds(
            cfg["cv"]["train_months"],
            cfg["cv"]["embargo_hours"],
            test_freq=cfg["cv"].get("test_freq", "M")     # 加這個
        )
    
    else:
        raise ValueError(f"Unknown fold type: {cv_type}")


def objective(trial: optuna.Trial, base_cfg: dict, df, run_dir: Path, pt_bundle=None) -> float:
    setup_cuda_acceleration()
    trial_dir = run_dir / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(base_cfg)
    cfg = suggest_rolling_and_cv(trial, cfg)
    device = "cuda" if (cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu"
    set_seed(int(cfg.get("seed", 42)) + trial.number)

    # ----- 抽超參數 -----
    cfg["train"]["lr"] = suggest_float(trial, "lr", cfg["train"]["lr"])
    cfg["train"]["weight_decay"] = suggest_float(trial, "weight_decay", cfg["train"]["weight_decay"])

    grad_clip = cfg["train"]["grad_clip"]
    cfg["train"]["grad_clip"] = trial.suggest_float("grad_clip", *grad_clip)

    # === 搜索 threshold（只限二分類）===
    thr_range = cfg["train"].get("threshold", None)
    if isinstance(thr_range, list) and len(thr_range) == 2:
        cfg["train"]["threshold"] = trial.suggest_float("threshold", thr_range[0], thr_range[1])
    else:
        cfg["train"]["threshold"] = float(thr_range) if thr_range is not None else None


    model_name = cfg["model"].get("name", "").lower()
    
    # model 架構參數
    # === LSTM 系列模型參數 ===
    if model_name in ["lstmhead", "lstm_se"]:
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
    
    # === TemporalTransformer 參數 ===
    elif model_name == "temporaltransformer":
        # 1. d_model
        d_model = trial.suggest_categorical("d_model", cfg["model"]["d_model"])
        
        # 2. n_heads
        n_heads = trial.suggest_categorical("n_heads", cfg["model"]["n_heads"])

        # ✅ 加入合法性檢查（避免 transformer 爆炸）
        if d_model % n_heads != 0:
            raise optuna.TrialPruned()

        cfg["model"]["d_model"] = d_model
        cfg["model"]["n_heads"] = n_heads
        
        # 3. n_layers
        n_layers = cfg["model"].get("n_layers", 2)
        if isinstance(n_layers, list) and len(n_layers) == 2:
            cfg["model"]["n_layers"] = trial.suggest_int("n_layers", int(n_layers[0]), int(n_layers[1]))
        else:
            cfg["model"]["n_layers"] = int(n_layers)

        # 4. mlp_ratio
        mlp_ratio = cfg["model"].get("mlp_ratio", 4.0)
        if isinstance(mlp_ratio, list) and len(mlp_ratio) == 2:
            cfg["model"]["mlp_ratio"] = trial.suggest_float("mlp_ratio", float(mlp_ratio[0]), float(mlp_ratio[1]))
        else:
            cfg["model"]["mlp_ratio"] = float(mlp_ratio)

        # dropout
        dp = cfg["model"].get("dropout", 0.0)
        if isinstance(dp, list) and len(dp) == 2:
            cfg["model"]["dropout"] = trial.suggest_float("dropout", dp[0], dp[1])
        else:
            cfg["model"]["dropout"] = float(dp)
            
        attdp = cfg["model"].get("attn_dropout", [0.0, 0.1])
        if isinstance(attdp, list) and len(attdp) == 2:
            cfg["model"]["attn_dropout"] = trial.suggest_float("attn_dropout", float(attdp[0]), float(attdp[1]))
        else:
            cfg["model"]["attn_dropout"] = float(attdp)






    # ----- 特徵抽樣 -----
    feat_pool = build_feature_pool(df.columns.to_list(), cfg)
    always_on = cfg.get("features", {}).get("always_on", [])
    blocklist = cfg["features"]["safety"]["blocklist"]
    k_range = cfg.get("features", {}).get("selection", {}).get("k_range", None)

    # 過濾 pool 裡的特徵（事前處理）
    feat_pool = [f for f in feat_pool if f not in blocklist]
    always_on = [f for f in always_on if f not in blocklist]

    selected_feats = sample_k_subset(trial, feat_pool, always_on, k_range)
    n_features = len(selected_feats)

    
    # selected_feats = [c for c in df.columns if c != "label" and c not in blocklist]
    # n_features = len(selected_feats)

    folds = folds_type(df, cfg)

    
    fold_scores = []
    for i, fold in enumerate(folds):
        tr_loader, va_loader, te_loader, _ = make_loaders_for_fold(df, selected_feats, "label", fold, cfg, cfg["cv"]["preload_gpu"])
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
        # === 選用 primary_metric ===
        primary = cfg["objective"].get("primary_metric", "macro_f1").lower()

        # if primary == "threshold_macro_f1":
        #     # 使用 threshold 調整後的 macro F1（僅限二分類）
        #     if "threshold_metrics" in result:
        #         score = float(result["threshold_metrics"]["macro_f1"])
        #     else:
        #         score = float(result.get("best_val_macro_f1", 0.0))  # fallback
        # elif primary == "macro_f1":
        #     score = float(result.get("best_val_macro_f1", 0.0))
        # elif primary == "macro_precision":
        #     score = float(result.get("test_metrics", {}).get("test_macro_precision", 0.0))
        # elif primary == "macro_recall":
        #     score = float(result.get("test_metrics", {}).get("test_macro_recall", 0.0))

        if primary == "threshold_macro_f05":
            if "threshold_metrics" in result and "f_05_macro" in result["threshold_metrics"]:
                score = float(result["threshold_metrics"]["f_05_macro"])
            # else:
            #     score = float(result.get("best_val_macro_f1", 0.0))  # fallback

        else:
            raise ValueError(f"Unknown primary_metric: {primary}")

        fold_scores.append(score)
        trial.report(score, step=i)


    mean_score  = float(np.mean(fold_scores)) if fold_scores else 0.0
    # mean_prec = float(np.mean(fold_scores)) 



    trial.set_user_attr("selected_features", selected_feats)
    trial.set_user_attr("n_features", n_features)
    trial.set_user_attr("fold_scores", fold_scores)

    import yaml
    trial_cfg_path = trial_dir / f"trial_config_{str(primary)}={mean_score}.yaml"
    with open(trial_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    return mean_score
