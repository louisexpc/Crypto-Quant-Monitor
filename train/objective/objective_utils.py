# objective_utils.py
import os, sys, re
import optuna
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from build_feature_loader.dataloader import FoldGenerator

# ======================================================================
# Section A. 小工具
# ======================================================================
def get_task_type(cfg: dict) -> str:
    """優先讀 cfg.task.type；否則看 num_classes 推斷。"""
    if "task" in cfg and "type" in cfg["task"]:
        return str(cfg["task"]["type"]).lower()
    if "target" in cfg and "type" in cfg["target"]:
        return str(cfg["target"]["type"]).lower()
    nc = int(cfg["model"].get("num_classes", 1))
    return "classification" if nc >= 2 else "regression"


def suggest_float(trial: optuna.Trial, name: str, val, log_names=frozenset(("lr","weight_decay"))):
    """支援：
       - 單值（直接回傳 float）
       - [low, high]（若 name 在 log_names 且皆>0，採 log 取樣）
       - [v1, v2, v3, ...]（categorical）
    """
    # 單值
    if isinstance(val, (int, float)):
        return float(val)
    # 1 個元素的 list
    if isinstance(val, list) and len(val) == 1:
        return float(val[0])
    # 區間
    if isinstance(val, list) and len(val) == 2:
        low, high = float(val[0]), float(val[1])
        use_log = (name in log_names) and (low > 0.0 and high > 0.0)
        return trial.suggest_float(name, low, high, log=use_log)
    # 多離散點
    if isinstance(val, list) and len(val) > 2:
        return trial.suggest_categorical(name, [float(v) for v in val])
    # 其他型別就原樣回傳
    return float(val)


def suggest_int(trial: optuna.Trial, name: str, val):
    """支援 [low, high] 或單值。"""
    if isinstance(val, (int, float)):
        return int(val)

    elif isinstance(val, list):
        if len(val) == 1:
            return int(val[0])
        elif len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
            return trial.suggest_int(name, int(val[0]), int(val[1]))
        elif len(val) > 2 and all(isinstance(x, (int, float)) for x in val):
            return trial.suggest_categorical(name, [int(v) for v in val])
    raise ValueError(f"[suggest_int] Unexpected value format: {val}")

def suggest_cat(trial: optuna.Trial, name: str, vals):
    """類別空間。"""
    if isinstance(vals, list) and len(vals) == 1:
        return vals[0]
    elif isinstance(vals, list) and len(vals) > 1:
        return trial.suggest_categorical(name, vals)
    else:
        return vals
    



# ======================================================================
# Section B. CV 與序列超參數（Rolling / Anchored / OddEven）
# ======================================================================
def suggest_rolling_and_cv(trial: optuna.Trial, cfg: dict) -> dict:
    """讓 Optuna 針對序列/切分做取樣；對於 list 皆採樣，單值則原樣。"""
    cv_cfg = cfg["cv"]
    cv_mode = cv_cfg["type"]

    # 1) 序列長度
    cfg["sequence"]["seq_len"] = suggest_int(trial, "sequence.seq_len", cfg["sequence"]["seq_len"])

    # 2) stride
    if "stride" in cfg["sequence"]:
        cfg["sequence"]["stride"] = suggest_int(trial, "sequence.stride", cfg["sequence"]["stride"])


    if cv_mode == "Purged_kfold":
        cv_cfg["n_splits"] = suggest_int(trial, "cv.n_splits", cv_cfg["n_splits"])
    
    elif cv_mode == "Rolling":
        # 4) Rolling 訓練窗（月數）
        cv_cfg["train_months"] = suggest_int(trial, "cv.train_months", cfg["cv"]["train_months"])

        # 5) 測試頻率（可為固定 or 多選）
        test_freq = cfg["cv"]["test_freq"]
        cv_cfg["test_freq"] = suggest_cat(trial, "cv.test_freq", test_freq) if isinstance(test_freq, list) else test_freq
    else:
        raise KeyError("no such mode")

    return cfg



def make_folds(df, cfg):
    """依 cfg.cv.type 產生 folds。"""    
    cfg_cv = cfg["cv"]
    cv_type = cfg_cv["type"]
    start_month = cfg["cv"]["start_date"] 
    fold_g = FoldGenerator(dt_index=df.index, mode=cv_type, start_month=start_month)
    

    if cv_type == "Purged_kfold":
        return fold_g.make_purged_kfold(
            n_splits=cfg_cv["n_splits"],
            embargo_hours=cfg_cv["embargo_hours"],
            min_train_days=cfg_cv["min_train_days"]
        )

    if cv_type == "Rolling":
        return fold_g.make_rolling_folds(
            train_window=cfg_cv["train_months"],
            embargo_hours=cfg_cv["embargo_hours"],
            test_freq=cfg_cv["test_freq"]
        )

    raise ValueError(f"Unknown fold type: {cv_type}")


# ======================================================================
# Section C. 模型超參數 Suggest（LSTM / TemporalTransformer）
# ======================================================================
def suggest_model_hparams(trial: optuna.Trial, cfg: dict) -> dict:
    model_name = str(cfg["model"].get("name", "")).lower()

    if model_name in ["lstm_se"]:
        for key in ["hidden_size", "n_layers"]:
            cfg["model"][key] = suggest_int(trial, key, cfg["model"][key])

        cfg["model"]["dropout"] = suggest_float(trial, "dropout", cfg["model"]["dropout"])
        cfg["model"]["bidirectional"] = False  # 固定單向

    elif model_name == "temporaltransformer":
        cfg["model"]["d_model"] = suggest_int(trial, "d_model", cfg["model"]["d_model"])
        cfg["model"]["n_heads"] = suggest_cat(trial, "n_heads", cfg["model"]["n_heads"])
        if cfg["model"]["d_model"] % cfg["model"]["n_heads"] != 0:
            raise optuna.TrialPruned()

        cfg["model"]["n_layers"] = suggest_int(trial, "n_layers", cfg["model"]["n_layers"])
        cfg["model"]["mlp_ratio"] = suggest_float(trial, "mlp_ratio", cfg["model"]["mlp_ratio"])
        cfg["model"]["dropout"] = suggest_float(trial, "dropout", cfg["model"]["dropout"])
        cfg["model"]["attn_dropout"] = suggest_float(trial, "attn_dropout", cfg["model"]["attn_dropout"])
        cfg["model"]["pooling"] = suggest_cat(trial, "pooling", cfg["model"]["pooling"])


    # ---- TwoStreamHybrid（雙流：minute LSTM + Transformer backbone）----
    if model_name == "twostreamhybrid":
        # backbone 與 TemporalTransformer 一致
        cfg["model"]["d_model"]      = suggest_int(trial, "model.d_model",      cfg["model"]["d_model"])
        cfg["model"]["n_heads"]      = suggest_cat(trial, "model.n_heads",      cfg["model"]["n_heads"])
        if cfg["model"]["d_model"] % cfg["model"]["n_heads"] != 0:
            raise optuna.TrialPruned()

        cfg["model"]["n_layers"]     = suggest_int(trial, "model.n_layers",     cfg["model"]["n_layers"])
        cfg["model"]["mlp_ratio"]    = suggest_float(trial, "model.mlp_ratio",  cfg["model"]["mlp_ratio"])
        cfg["model"]["dropout"]      = suggest_float(trial, "model.dropout",    cfg["model"]["dropout"])
        cfg["model"]["attn_dropout"] = suggest_float(trial, "model.attn_dropout", cfg["model"]["attn_dropout"])
        cfg["model"]["pooling"]      = suggest_cat(trial, "model.pooling",      cfg["model"]["pooling"])
        # minute-LSTM 分支
        # minute_steps 通常是資料規格（例如 15），除非你真的要搜，否則視為單值
        if "minute_hidden" in cfg["model"]:
            cfg["model"]["minute_hidden"]  = suggest_int(trial, "model.minute_hidden",  cfg["model"]["minute_hidden"])
        if "minute_layers" in cfg["model"]:
            cfg["model"]["minute_layers"]  = suggest_int(trial, "model.minute_layers",  cfg["model"]["minute_layers"])
        if "minute_dropout" in cfg["model"]:
            cfg["model"]["minute_dropout"] = suggest_float(trial, "model.minute_dropout", cfg["model"]["minute_dropout"])


    # 預設分類/回歸的 num_classes
    task_type = get_task_type(cfg)
    if task_type == "classification":
        cfg["model"]["num_classes"] = int(cfg["model"].get("num_classes", 2))
    else:
        cfg["model"]["num_classes"] = int(cfg["model"].get("num_classes", 1))

    return cfg


# ======================================================================
# Section D. 特徵池
# ======================================================================

# def get_enabled_feature_names(cfg: dict, df_columns: list[str], target_col: str | None = None) -> list[str]:
#     if target_col is None:
#         target_col = "label" if str(cfg["task"]["type"]).lower() == "classification" else "target"
#     blacklist = {target_col, "label", "target", "y", "y_cls", "y_reg"}
#     return [c for c in df_columns if c not in blacklist]




# ======================================================================
# Section E. 安全組 tag、重新命名 trial 目錄
# ======================================================================

def _format_score_tag(name: str, val, digits: int = 4, signed: bool = True) -> str:
    """回傳像 'pearson=+0.1234' 的 tag；若不是有限數字則回傳 'pearson=nan'。"""
    v = float(val) if isinstance(val, (int, float, np.floating)) else np.nan
    if not np.isfinite(v):
        return f"{name}=nan"
    sign = "+" if (signed and v >= 0) else ""
    return f"{name}={sign}{v:.{digits}f}"

def _safe_rename_trial_dir(trial_dir: Path, tags: list[str]) -> Path:
    """將 trial 目錄重新命名為 自動避開重名。"""
    parent = trial_dir.parent
    base = trial_dir.name
    # 清理不安全字元
    safe_tags = [re.sub(r"[^a-zA-Z0-9_=+\-\.]", "", t) for t in tags if t]
    new_name = f"{base}_{'_'.join(safe_tags)}" if safe_tags else base
    new_dir = parent / new_name
    idx = 1
    while new_dir.exists():
        new_dir = parent / f"{new_name}__{idx}"
        idx += 1
    try:
        os.rename(trial_dir, new_dir)
        print(f"[objective] Renamed trial dir → {new_dir.name}")
        return new_dir
    except Exception as e:
        print(f"[objective] Rename failed: {e}. Keep original: {trial_dir}")
        return trial_dir