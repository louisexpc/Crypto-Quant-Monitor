# objective_runtime.py
"""
Runtime 版本的 Optuna 目標函式（模組化重構）：
- 假設 df 已包含：Indicators 計算出的特徵 (已 shift(1) 防洩漏) + 'label'（若為分類）
- 不再做預算特徵與安全黑名單等處理
- 可選：在已計算的特徵中，再讓 Optuna 抽一個子集 (features.selection.k_range)
- 同時支援 classification / regression，並依 cfg.objective.primary_metric / direction 回傳分數
"""

from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
import optuna
import yaml

# --- Trainer / Data / Model ---
# from trainer import train_one_fold            # ★ 使用新版 trainer
from trainer.trainer_base import get_trainer
from utils.dataloader import FoldGenerator, make_loaders_for_fold
from models.model_factory import build_model
from utils.init_train import set_seed, setup_cuda_acceleration, sample_k_subset


# ======================================================================
# Section A. 小工具：任務類型 / Optuna 取值
# ======================================================================
def get_task_type(cfg: dict) -> str:
    """優先讀 cfg.task.type；否則看 num_classes 推斷。"""
    if "task" in cfg and "type" in cfg["task"]:
        return str(cfg["task"]["type"]).lower()
    if "target" in cfg and "type" in cfg["target"]:
        return str(cfg["target"]["type"]).lower()
    nc = int(cfg["model"].get("num_classes", 1))
    return "classification" if nc >= 2 else "regression"


def suggest_float(trial: optuna.Trial, name: str, val, log_names=frozenset(("lr", "weight_decay"))):
    """支援 [low, high]（可 log）、或單值直接轉 float。"""
    if isinstance(val, list) and len(val) == 2:
        low, high = float(val[0]), float(val[1])
        use_log = (name in log_names) and (low > 0.0 and high > 0.0)
        return trial.suggest_float(name, low, high, log=use_log)
    return float(val)


def suggest_int(trial: optuna.Trial, name: str, val):
    """支援 [low, high] 或單值。"""
    if isinstance(val, list) and len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
        return int(trial.suggest_int(name, int(val[0]), int(val[1])))
    return int(val)


def suggest_cat(trial: optuna.Trial, name: str, vals):
    """類別空間。"""
    return trial.suggest_categorical(name, tuple(vals))


# ======================================================================
# Section B. CV 與序列超參數（Rolling / Anchored / OddEven）
# ======================================================================
def suggest_rolling_and_cv(trial: optuna.Trial, cfg: dict) -> dict:
    """讓 Optuna 針對序列長度 / Rolling 相關做取樣；維持與你的設定相容。"""
    # 1) seq_len
    seq_len_cfg = cfg["sequence"]["seq_len"]
    if isinstance(seq_len_cfg, list):
        seq_len = trial.suggest_categorical("seq_len", [int(x) for x in seq_len_cfg])
    else:
        seq_len = int(seq_len_cfg)
    cfg["sequence"]["seq_len"] = seq_len

    # 2) Rolling 訓練窗（月數）
    if isinstance(cfg["cv"]["train_months"], list):
        train_months = suggest_cat(trial, "cv.train_months", cfg["cv"]["train_months"])
    else:
        train_months = int(cfg["cv"]["train_months"])
    cfg["cv"]["train_months"] = int(train_months)

    # 3) 測試頻率
    if isinstance(cfg["cv"]["test_freq"], list):
        test_freq = suggest_cat(trial, "cv.test_freq", cfg["cv"]["test_freq"])
    else:
        test_freq = cfg["cv"]["test_freq"]
    cfg["cv"]["test_freq"] = test_freq

    # 4) embargo（避免序列重疊）
    bar_size_hours = 1  # 你的資料為 1h bar；若是 15m 可改 0.25
    embargo_min = max(0, (seq_len - 1) * bar_size_hours)
    emb_cfg = cfg["cv"].get("embargo_hours", [24, 48])
    embargo_max_cfg = int(emb_cfg[1]) if isinstance(emb_cfg, list) else int(emb_cfg)
    embargo = trial.suggest_int("cv.embargo_hours", int(embargo_min), int(embargo_max_cfg), step=1)
    cfg["cv"]["embargo_hours"] = int(embargo)

    # 5) train/val split
    split_low, split_high = cfg["cv"]["train_val_split"]
    tv_split = trial.suggest_float("cv.train_val_split", float(split_low), float(split_high))
    cfg["cv"]["train_val_split"] = float(tv_split)
    return cfg


def make_folds(df, cfg):
    """依 cfg.cv.type 產生 folds。"""
    cv_type = cfg["cv"]["type"]
    start_month = cfg["cv"].get("start_month", str(df.index[0].date()))
    fold_g = FoldGenerator(dt_index=df.index, mode=cv_type, start_month=start_month)

    if cv_type == "OddEven":
        return fold_g.make_two_month_folds()

    if cv_type == "Anchored":
        return fold_g.make_anchored_folds(
            embargo_hours=cfg["cv"].get("embargo_hours", 24),
            min_train_days=cfg["cv"].get("min_train_days", 30),
            test_freq=cfg["cv"].get("test_freq", "M")
        )

    if cv_type == "Rolling":
        return fold_g.make_rolling_folds(
            cfg["cv"]["train_months"],
            cfg["cv"]["embargo_hours"],
            test_freq=cfg["cv"].get("test_freq", "M")
        )

    raise ValueError(f"Unknown fold type: {cv_type}")


# ======================================================================
# Section C. 模型超參數 Suggest（LSTM / TemporalTransformer）
# ======================================================================
def suggest_model_hparams(trial: optuna.Trial, cfg: dict) -> dict:
    model_name = str(cfg["model"].get("name", "")).lower()

    if model_name in ["lstmhead", "lstm_se"]:
        # hidden_size, n_layers
        for key in ["hidden_size", "n_layers"]:
            val = cfg["model"][key]
            if isinstance(val, list):
                if len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
                    cfg["model"][key] = trial.suggest_int(key, int(val[0]), int(val[1]))
                else:
                    cfg["model"][key] = suggest_cat(trial, key, val)
            else:
                cfg["model"][key] = int(val)
        # dropout
        dp = cfg["model"].get("dropout", 0.0)
        if isinstance(dp, list) and len(dp) == 2:
            cfg["model"]["dropout"] = trial.suggest_float("dropout", float(dp[0]), float(dp[1]))
        elif isinstance(dp, list):
            cfg["model"]["dropout"] = trial.suggest_categorical("dropout", [float(x) for x in dp])
        else:
            cfg["model"]["dropout"] = float(dp)
        cfg["model"]["bidirectional"] = False  # 固定單向

    elif model_name == "temporaltransformer":
        # d_model / n_heads
        d_model = trial.suggest_categorical("d_model", cfg["model"]["d_model"])
        n_heads = trial.suggest_categorical("n_heads", cfg["model"]["n_heads"])
        if int(d_model) % int(n_heads) != 0:
            raise optuna.TrialPruned()  # 防呆
        cfg["model"]["d_model"] = int(d_model)
        cfg["model"]["n_heads"] = int(n_heads)

        # n_layers
        n_layers = cfg["model"].get("n_layers", 2)
        if isinstance(n_layers, list) and len(n_layers) == 2:
            cfg["model"]["n_layers"] = trial.suggest_int("n_layers", int(n_layers[0]), int(n_layers[1]))
        elif isinstance(n_layers, list):
            cfg["model"]["n_layers"] = trial.suggest_categorical("n_layers", [int(x) for x in n_layers])
        else:
            cfg["model"]["n_layers"] = int(n_layers)

        # mlp_ratio
        mlp_ratio = cfg["model"].get("mlp_ratio", 4.0)
        if isinstance(mlp_ratio, list) and len(mlp_ratio) == 2:
            cfg["model"]["mlp_ratio"] = trial.suggest_float("mlp_ratio", float(mlp_ratio[0]), float(mlp_ratio[1]))
        elif isinstance(mlp_ratio, list):
            cfg["model"]["mlp_ratio"] = trial.suggest_categorical("mlp_ratio", [float(x) for x in mlp_ratio])
        else:
            cfg["model"]["mlp_ratio"] = float(mlp_ratio)

        # dropout
        dp = cfg["model"].get("dropout", 0.0)
        if isinstance(dp, list) and len(dp) == 2:
            cfg["model"]["dropout"] = trial.suggest_float("dropout", float(dp[0]), float(dp[1]))
        elif isinstance(dp, list):
            cfg["model"]["dropout"] = trial.suggest_categorical("dropout", [float(x) for x in dp])
        else:
            cfg["model"]["dropout"] = float(dp)

        # attn_dropout
        attdp = cfg["model"].get("attn_dropout", [0.0, 0.1])
        if isinstance(attdp, list) and len(attdp) == 2:
            cfg["model"]["attn_dropout"] = trial.suggest_float("attn_dropout", float(attdp[0]), float(attdp[1]))
        elif isinstance(attdp, list):
            cfg["model"]["attn_dropout"] = trial.suggest_categorical("attn_dropout", [float(x) for x in attdp])
        else:
            cfg["model"]["attn_dropout"] = float(attdp)

        # pooling
        pooling = cfg["model"].get("pooling", "attn")
        if isinstance(pooling, list):
            cfg["model"]["pooling"] = trial.suggest_categorical("pooling", tuple(pooling))
        else:
            cfg["model"]["pooling"] = str(pooling)

            # ★★★ 回歸時，強制 attn，比較不會退化成常數 ★★★
        if get_task_type(cfg) == "regression":
            cfg["model"]["pooling"] = "attn"      # <—— 加這行


        # flags
        def _suggest_bool(key, default=False):
            val = cfg["model"].get(key, default)
            if isinstance(val, list):
                return bool(trial.suggest_categorical(key, [bool(x) for x in val]))
            return bool(val)

        cfg["model"]["use_causal"]      = _suggest_bool("use_causal", True)
        cfg["model"]["use_alibi"]       = _suggest_bool("use_alibi", True)
        cfg["model"]["use_conv_stem"]   = _suggest_bool("use_conv_stem", True)
        cfg["model"]["use_input_norm"]  = _suggest_bool("use_input_norm", True)
        cfg["model"]["use_learned_pos"] = _suggest_bool("use_learned_pos", False)

        # alibi_slope
        alibi = cfg["model"].get("alibi_slope", 0.05)
        if isinstance(alibi, list) and len(alibi) == 2:
            cfg["model"]["alibi_slope"] = trial.suggest_float("alibi_slope", float(alibi[0]), float(alibi[1]))
        elif isinstance(alibi, list):
            cfg["model"]["alibi_slope"] = trial.suggest_categorical("alibi_slope", [float(x) for x in alibi])
        else:
            cfg["model"]["alibi_slope"] = float(alibi)

        # droppath
        dp_path = cfg["model"].get("droppath", 0.0)
        if isinstance(dp_path, list) and len(dp_path) == 2:
            cfg["model"]["droppath"] = trial.suggest_float("droppath", float(dp_path[0]), float(dp_path[1]))
        elif isinstance(dp_path, list):
            cfg["model"]["droppath"] = trial.suggest_categorical("droppath", [float(x) for x in dp_path])
        else:
            cfg["model"]["droppath"] = float(dp_path)

    # 預防 num_classes 遺漏：依任務型態填合理預設
    task_type = get_task_type(cfg)
    if task_type == "classification":
        cfg["model"]["num_classes"] = int(cfg["model"].get("num_classes", 2))
    else:
        cfg["model"]["num_classes"] = int(cfg["model"].get("num_classes", 1))
    return cfg


# ======================================================================
# Section D. 特徵池與子集抽樣（選配）
# ======================================================================
def build_feature_pool(df, cfg, trial: optuna.Trial | None, target_col: str):
    """從 df 欄位建立特徵池；若 cfg.features.selection.k_range 存在，則抽子集。"""
    feat_pool = [c for c in df.columns if c != target_col]

    sel_cfg   = (cfg.get("features", {}) or {}).get("selection", {})
    k_range   = sel_cfg.get("k_range", None)          # 例如 [64, 256]
    always_on = sel_cfg.get("always_on", [])          # 例如 ["RSI_14", "MACD_12_26_9"]

    # 僅當提供 k_range 時才做子集抽樣；抽樣由 sample_k_subset 內部完成（含 k 和 seed 的 suggest）
    if trial is not None and k_range and isinstance(k_range, (list, tuple)) and len(k_range) == 2:
        k_min, k_max = int(k_range[0]), int(k_range[1])
        feat_pool = sample_k_subset(
            trial=trial,
            pool=feat_pool,
            always_on=list(always_on),
            k_range=(k_min, k_max)
        )
    return feat_pool


# ======================================================================
# Section E. Trial 得分計算（依 primary_metric / direction）
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
# Section F. 主目標函式（給 Optuna）
# ======================================================================
def objective(trial: optuna.Trial, base_cfg: dict, df, run_dir: Path, pt_bundle=None) -> float:
    """
    df: 若為分類需包含 'label' 欄位（0/1），以及多個 runtime 特徵欄位。
    run_dir: 這個 trial 的輸出根目錄。
    備註：
      - 會依 cfg.objective.{primary_metric, direction} 返還分數；
        若你的 study 是 direction='minimize'，請在 YAML 內對齊，或讓外部建立 study 時使用相同方向。
    """
    setup_cuda_acceleration()
    cfg = copy.deepcopy(base_cfg)             # <--- 先複製
    task_type = get_task_type(cfg)            # <--- 再判斷任務
    target_col = "label" if task_type == "classification" else "target"

    trial_dir = run_dir / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cfg = suggest_rolling_and_cv(trial, cfg)

    # ---- 同步 train_val_split 到 cfg.data ----
    cfg.setdefault("data", {})
    cfg["data"]["train_val_split"] = float(cfg["cv"]["train_val_split"])

    device = "cuda" if (cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu"
    set_seed(int(cfg.get("seed", 42)) + trial.number)


    # ---- 訓練超參 ----
    cfg["train"]["lr"] = suggest_float(trial, "lr", cfg["train"]["lr"])
    cfg["train"]["weight_decay"] = suggest_float(trial, "weight_decay", cfg["train"]["weight_decay"])
    # cfg["train"]["batch_size"] = suggest_cat(trial, "batch_size", cfg["train"]["batch_size"])
    cfg["train"]["batch_size"] = cfg["train"]["batch_size"]

    grad_clip = cfg["train"]["grad_clip"]
    if isinstance(grad_clip, list) and len(grad_clip) == 2:
        cfg["train"]["grad_clip"] = trial.suggest_float("grad_clip", float(grad_clip[0]), float(grad_clip[1]))
    else:
        cfg["train"]["grad_clip"] = float(grad_clip)

    # 二分類 threshold（僅在 classification 下有意義）
    thr_range = cfg["train"].get("threshold", None)
    if isinstance(thr_range, list) and len(thr_range) == 2:
        cfg["train"]["threshold"] = trial.suggest_float("threshold", float(thr_range[0]), float(thr_range[1]))
    elif thr_range is not None:
        cfg["train"]["threshold"] = float(thr_range)
    else:
        cfg["train"]["threshold"] = None

    # ---- 損失/目標參數（回歸用）----
    if task_type == "regression":
        for key, space in [("alpha", cfg["loss"].get("alpha", 0.7)),
                           ("ema_decay", cfg["loss"].get("ema_decay", 0.9))]:
            if isinstance(space, list) and len(space) == 2:
                cfg["loss"][key] = trial.suggest_float(f"loss.{key}", float(space[0]), float(space[1]))
            else:
                cfg["loss"][key] = float(space)

        beta = cfg["loss"].get("beta", None)
        if beta is None:
            cfg["loss"]["beta"] = float(max(0.0, 1.0 - float(cfg["loss"]["alpha"])))
        elif isinstance(beta, list) and len(beta) == 2:
            cfg["loss"]["beta"] = trial.suggest_float("loss.beta", float(beta[0]), float(beta[1]))
        else:
            cfg["loss"]["beta"] = float(beta)

    # ---- 模型超參 ----
    cfg = suggest_model_hparams(trial, cfg)

    # ---- 特徵池（可選子集抽樣）----
    feat_pool = build_feature_pool(df, cfg, trial, target_col)
    n_features = len(feat_pool)

    # ---- 產生 folds ----
    folds = make_folds(df, cfg)

    # ★ 根據任務載入對應 trainer
    train_one_fold = get_trainer(cfg)

    # ---- 逐 fold 訓練 ----
    fold_scores = []
    for i, fold in enumerate(folds):
        tr_loader, va_loader, te_loader, _ = make_loaders_for_fold(
            df, feat_pool, target_col, fold, cfg, cfg["cv"].get("preload_gpu", False)
        )
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

        score = compute_trial_score(result, cfg)  # ★ 依 primary/direction 自動出分
        fold_scores.append(float(score))
        trial.report(float(score), step=i)
        # if trial.should_prune():
        #     raise optuna.TrialPruned()

    # ---- 加總/紀錄 ----
    mean_score = float(np.mean(fold_scores))

    trial.set_user_attr("selected_features", feat_pool)
    trial.set_user_attr("n_features", n_features)
    trial.set_user_attr("fold_scores", fold_scores)
    trial.set_user_attr("task_type", task_type)            # ← 保留一個即可
    trial.set_user_attr("primary_metric", str(cfg["objective"].get("primary_metric", "")))
    trial.set_user_attr("direction", str(cfg["objective"].get("direction", "")))
    trial.set_user_attr("target_col", target_col)

    trial_cfg_path = trial_dir / f"trial_config_{str(cfg['objective'].get('primary_metric',''))}={mean_score:.6g}.yaml"
    with open(trial_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    return mean_score