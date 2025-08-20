# objective_utils.py
import optuna
import os
import sys
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


def suggest_float(trial: optuna.Trial, name: str, val, log_names=frozenset(("lr", "weight_decay"))):
    """支援 [low, high]（可 log）、或單值直接轉 float。"""
    if isinstance(val,(int, float)) or len(val) == 1:
        return float(val)

    elif isinstance(val, list) and len(val) == 2:
        low, high = float(val[0]), float(val[1])
        use_log = (name in log_names) and (low > 0.0 and high > 0.0)
        return trial.suggest_float(name, low, high, log=use_log)
    
    return trial.suggest_categorical(name, [float(v) for v in val])


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
    """讓 Optuna 針對序列長度 / Rolling 相關做取樣；維持與你的設定相容。"""

    # 1) 序列長度（要先取，後續 embargo 要用）
    seq_len = suggest_int(trial, "sequence.seq_len", cfg["sequence"]["seq_len"])
    cfg["sequence"]["seq_len"] = seq_len

    # 2) Rolling 訓練窗（月數）
    cfg["cv"]["train_months"] = suggest_int(trial, "cv.train_months", cfg["cv"]["train_months"])

    # 3) 測試頻率（可為固定 or 多選）
    test_freq = cfg["cv"]["test_freq"]
    cfg["cv"]["test_freq"] = (
        suggest_cat(trial, "cv.test_freq", test_freq) if isinstance(test_freq, list) else test_freq
    )

    # 4) embargo 時數（強制下限 ≥ seq_len）
    embargo_range = cfg["cv"]["embargo_hours"]  # 若沒設會報錯
    assert isinstance(embargo_range, (list, tuple)) and len(embargo_range) == 2, \
        f"embargo_hours 應為 [min, max] 形式，目前為：{embargo_range}"

    embargo_min = max(int(embargo_range[0]), seq_len - 1)  # 避免序列重疊
    embargo_max = int(embargo_range[1])
    cfg["cv"]["embargo_hours"] = trial.suggest_int("cv.embargo_hours", embargo_min, embargo_max)

    # 5) train/val split
    cfg["cv"]["train_val_split"] = suggest_float(trial, "cv.train_val_split", cfg["cv"]["train_val_split"])

    return cfg


def make_folds(df, cfg):
    """依 cfg.cv.type 產生 folds。"""
    cv_type = cfg["cv"]["type"]
    start_month = cfg["cv"]["start_date"] 
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

    # 預設分類/回歸的 num_classes
    task_type = get_task_type(cfg)
    if task_type == "classification":
        cfg["model"]["num_classes"] = int(cfg["model"].get("num_classes", 2))
    else:
        cfg["model"]["num_classes"] = int(cfg["model"].get("num_classes", 1))

    return cfg


# ======================================================================
# Section D. 特徵池與子集抽樣（選配）
# ======================================================================
def _prune_plan(plan: dict) -> dict:
    """
    回傳 enabled=True 的特徵子集
    """
    feats = plan.get("features", [])
    feats_enabled = [f for f in feats if f.get("enabled", True)]
    return {"features": feats_enabled}


import re
from typing import List

# def map_feature_to_columns(full_name: str, df_columns: list[str]) -> List[str]:
#     """
#     根據 full_name 推估對應的 df.columns 欄位，支援常見 pandas-ta 命名規則。
#     """
#     matched_cols = []

#     # === 解析名稱與參數 ===
#     m = re.match(r"(\w+)\((.*)\)", full_name)
#     if m:
#         name, args_str = m.group(1), m.group(2)
#         args = dict([kv.split("=") for kv in args_str.split(",")])
#     else:
#         name, args = full_name, {}

#     name = name.upper()

#     # === 根據規則產生預測欄位名稱 ===
#     def exists(col):
#         return col in df_columns

#     if name == "RSI":
#         col = f"RSI_{args['length']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "MACD":
#         col = f"MACD_{args['fast']}_{args['slow']}_{args['signal']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "MOM":
#         col = f"MOM_{args['length']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "LOGRET":
#         col = f"LOGRET_{args['length']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "STOCH":
#         k = args['k']
#         if exists(f"STOCHk_{k}"): matched_cols.append(f"STOCHk_{k}")
#         if exists(f"STOCHd_{k}"): matched_cols.append(f"STOCHd_{k}")

#     elif name == "KDJ":
#         k, d = args['k'], args['d']
#         if exists(f"J_{k}_{d}"): matched_cols.append(f"J_{k}_{d}")

#     elif name == "RVI":
#         l = args['length']
#         if exists(f"RVI_{l}"): matched_cols.append(f"RVI_{l}")

#     elif name == "WILLR":
#         l = args['length']
#         if exists(f"WILLR_{l}"): matched_cols.append(f"WILLR_{l}")

#     elif name == "UO":
#         col = f"UO_{args['fast']}_{args['medium']}_{args['slow']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "DPO":
#         l = args['length']
#         if exists(f"DPO_{l}"): matched_cols.append(f"DPO_{l}")

#     elif name == "CCI":
#         col = f"CCI_{args['length']}_{args['c']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "BBP":
#         col = f"BBP_{args['length']}_{args['std']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "ZS":
#         col = f"ZS_{args['length']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "SLOPE":
#         col = f"SLOPE_{args['length']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "EBSW":
#         col = f"EBSW_{args['length']}_{args['mamode']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "MASSI":
#         col = f"MASSI_{args['fast']}_{args['slow']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "PVO":
#         col = f"PVO_{args['fast']}_{args['slow']}_{args['signal']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "TRUERANGE":
#         if exists("TRUERANGE_1"): matched_cols.append("TRUERANGE_1")

#     elif name == "AMATE_LR":
#         col = f"AMATe_LR_{args['fast']}_{args['slow']}_{args['mamode']}"
#         if exists(col): matched_cols.append(col)

#     elif name == "BOP":
#         if exists("BOP"): matched_cols.append("BOP")

#     else:
#         raise KeyError(f"❌ 無法推測特徵 '{full_name}' 的欄位名稱（未定義規則）")

#     if not matched_cols:
#         raise ValueError(f"❌ 特徵 '{full_name}' 無對應欄位存在於 df.columns")

#     return matched_cols


import re
from typing import List

def map_feature_to_columns(full_name: str, df_columns: list[str]) -> List[str]:
    """
    根據 full_name 推估對應的 df.columns 欄位，支援 pandas-ta 與自定義命名規則。
    """
    matched_cols = []

    # === 解析名稱與參數 ===
    m = re.match(r"(\w+)\((.*)\)", full_name)
    if m:
        name, args_str = m.group(1), m.group(2)
        args = dict([kv.split("=") for kv in args_str.split(",")])
    else:
        name, args = full_name, {}

    name = name.upper()

    def exists(col): return col in df_columns

    match name:
        # === 常見技術指標 ===
        case "RSI":
            col = f"RSI_{args['length']}"
            if exists(col): matched_cols.append(col)

        case "MACD":
            col = f"MACD_{args['fast']}_{args['slow']}_{args['signal']}"
            if exists(col): matched_cols.append(col)

        case "MOM":
            col = f"MOM_{args['length']}"
            if exists(col): matched_cols.append(col)

        case "LOGRET":
            col = f"LOGRET_{args['length']}"
            if exists(col): matched_cols.append(col)

        case "STOCH":
            k = args['k']
            if exists(f"STOCHk_{k}"): matched_cols.append(f"STOCHk_{k}")
            if exists(f"STOCHd_{k}"): matched_cols.append(f"STOCHd_{k}")

        case "KDJ":
            k, d = args['k'], args['d']
            if exists(f"J_{k}_{d}"): matched_cols.append(f"J_{k}_{d}")

        case "RVI":
            l = args['length']
            if exists(f"RVI_{l}"): matched_cols.append(f"RVI_{l}")

        case "WILLR":
            l = args['length']
            if exists(f"WILLR_{l}"): matched_cols.append(f"WILLR_{l}")

        case "UO":
            col = f"UO_{args['fast']}_{args['medium']}_{args['slow']}"
            if exists(col): matched_cols.append(col)

        case "DPO":
            l = args['length']
            if exists(f"DPO_{l}"): matched_cols.append(f"DPO_{l}")

        case "CCI":
            col = f"CCI_{args['length']}_{args['c']}"
            if exists(col): matched_cols.append(col)

        case "BBP":
            col = f"BBP_{args['length']}_{args['std']}"
            if exists(col): matched_cols.append(col)

        case "ZS":
            col = f"ZS_{args['length']}"
            if exists(col): matched_cols.append(col)

        case "SLOPE":
            col = f"SLOPE_{args['length']}"
            if exists(col): matched_cols.append(col)

        case "EBSW":
            col = f"EBSW_{args['length']}_{args['mamode']}"
            if exists(col): matched_cols.append(col)

        case "MASSI":
            col = f"MASSI_{args['fast']}_{args['slow']}"
            if exists(col): matched_cols.append(col)

        case "PVO":
            col = f"PVO_{args['fast']}_{args['slow']}_{args['signal']}"
            if exists(col): matched_cols.append(col)

        case "TRUERANGE":
            if exists("TRUERANGE_1"): matched_cols.append("TRUERANGE_1")

        case "AMATE_LR":
            col = f"AMATe_LR_{args['fast']}_{args['slow']}_{args['mamode']}"
            if exists(col): matched_cols.append(col)

        case "BOP":
            if exists("BOP"): matched_cols.append("BOP")

        # === 原始 OHLCV 欄位 ===
        case "OPEN" | "HIGH" | "LOW" | "CLOSE" | "VOLUME":
            if exists(name.lower()): matched_cols.append(name.lower())

        case _:
            raise KeyError(f"❌ 無法推測特徵 '{full_name}' 的欄位名稱（未定義規則）")

    if not matched_cols:
        raise ValueError(f"❌ 特徵 '{full_name}' 無對應欄位存在於 df.columns，請檢查拼字或 plan.yaml")

    return matched_cols

def get_enabled_feature_names(cfg: dict, df_columns: list[str]) -> list[str]:
    feat_plan = cfg["features"]["plan"]["features"]
    feat_pool = []

    for item in feat_plan:
        if not item.get("enabled", True):
            continue

        name = item["name"].upper()
        kwargs = item.get("kwargs", {})
        args_repr = ",".join(f"{k}={v}" for k, v in kwargs.items())
        full_name = f"{name}({args_repr})" if args_repr else name

        # 使用動態 map
        cols = map_feature_to_columns(full_name, df_columns)
        feat_pool.extend(cols)

    return feat_pool