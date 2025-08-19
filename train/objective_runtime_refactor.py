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
# Section A. 小工具：任務類型 / 通用 Optuna 取值器
# ======================================================================
def get_task_type(cfg: dict) -> str:
    if "task" in cfg and "type" in cfg["task"]:
        return str(cfg["task"]["type"]).lower()
    if "target" in cfg and "type" in cfg["target"]:
        return str(cfg["target"]["type"]).lower()
    nc = int(cfg["model"].get("num_classes", 1))
    return "classification" if nc >= 2 else "regression"


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _is_range(space) -> bool:
    return isinstance(space, (list, tuple)) and len(space) == 2 and all(_is_number(v) for v in space)

def _infer_numeric_type(space, type_hint: str | None) -> str:
    """
    回傳 'int' 或 'float'
    - 若明確給 type_hint 就用
    - 否則：純整數區間 → int；其他 → float
    """
    if type_hint in ("int", "float"):
        return type_hint
    if _is_range(space):
        lo, hi = space
        if float(lo).is_integer() and float(hi).is_integer():
            return "int"
    return "float"

def suggest_any(
    trial: optuna.Trial,
    name: str,
    space,
    *,
    type_hint: str | None = None,     # 'int'|'float'|'bool'|'cat'|'auto'
    log: bool | None = None,          # 對數域（只對 float 有效）
    step: int | float | None = None,  # 對 int/float 的步進
    dynamic_low: float | None = None, # 動態下限（例如依 seq_len 推導的 embargo_min）
    dynamic_high: float | None = None # 動態上限
):
    """
    統一規則：
    - 單值：直接回傳（型別轉換依 type_hint）
    - 區間(list/tuple, len=2, 皆為數字)：uniform 抽樣（自動判斷 int/float；可給 log/step）
    - 列舉(list/tuple, len>=1, 非區間 或 含非數字)：categorical
    - dict: 支援 {'low','high','choices','type','log','step'} 混搭
    """
    # None 直接回 None（允許某些可選參數）
    if space is None:
        return None

    # dict 空間：標準化成上面的格式
    if isinstance(space, dict):
        if "choices" in space:
            return trial.suggest_categorical(name, tuple(space["choices"]))
        low = space.get("low", None)
        high = space.get("high", None)
        if low is not None and high is not None:
            th = space.get("type", type_hint)
            return suggest_any(
                trial, name, [low, high],
                type_hint=th,
                log=space.get("log", log),
                step=space.get("step", step),
                dynamic_low=dynamic_low, dynamic_high=dynamic_high
            )
        # 單值 dict（例如 {'value': 128}）
        if "value" in space:
            val = space["value"]
            if type_hint == "int":   return int(val)
            if type_hint == "float": return float(val)
            if type_hint == "bool":  return bool(val)
            return val
        # 其他情況：丟回 categorical
        return trial.suggest_categorical(name, tuple(space))

    # 非 dict：處理 list/tuple/標量
    # 區間：兩元素且數字
    if _is_range(space):
        lo, hi = float(space[0]), float(space[1])
        if dynamic_low is not None:  lo = max(lo, float(dynamic_low))
        if dynamic_high is not None: hi = min(hi, float(dynamic_high))
        kind = _infer_numeric_type(space, type_hint)
        if kind == "int":
            st = int(step) if step is not None else 1
            return int(trial.suggest_int(name, int(np.floor(lo)), int(np.ceil(hi)), step=st))
        else:
            lg = bool(log) if log is not None else False
            return float(trial.suggest_float(name, lo, hi, log=lg))

    # 列舉：list/tuple（不是區間）→ categorical
    if isinstance(space, (list, tuple)):
        return trial.suggest_categorical(name, tuple(space))

    # 單值：直接回傳並做必要型別轉換
    if type_hint == "int":
        return int(space)
    if type_hint == "float":
        return float(space)
    if type_hint == "bool":
        return bool(space)
    return space

# ======================================================================
# Section B. CV 與序列超參數（Rolling / Anchored / OddEven）
# ======================================================================
def suggest_rolling_and_cv(trial: optuna.Trial, cfg: dict) -> dict:
    # 1) seq_len（可單值 / 離散集合）
    cfg["sequence"]["seq_len"] = int(
        suggest_any(trial, "seq_len", cfg["sequence"]["seq_len"], type_hint="int")
    )

    # 2) Rolling 訓練窗（月數）：單值 or 列舉
    cfg["cv"]["train_months"] = int(
        suggest_any(trial, "cv.train_months", cfg["cv"]["train_months"], type_hint="int")
    )

    # 3) 測試頻率：單值 or 列舉（例如 ["M","2M","Q"]）
    cfg["cv"]["test_freq"] = str(
        suggest_any(trial, "cv.test_freq", cfg["cv"]["test_freq"], type_hint="cat")
    )

    # 4) embargo（避免序列重疊）：下限由 seq_len 決定
    bar_size_hours = 1  # 你的資料是 1H bar；若 15m 可設 0.25
    embargo_min = max(0, (cfg["sequence"]["seq_len"] - 1) * bar_size_hours)
    # cfg["cv"]["embargo_hours"] 可為單值或 [low,high]
    cfg["cv"]["embargo_hours"] = int(
        suggest_any(
            trial, "cv.embargo_hours", cfg["cv"].get("embargo_hours", [24, 48]),
            type_hint="int", step=1, dynamic_low=embargo_min
        )
    )

    # 5) train/val split: 區間/單值皆可
    cfg["cv"]["train_val_split"] = float(
        suggest_any(trial, "cv.train_val_split", cfg["cv"]["train_val_split"], type_hint="float")
    )
    return cfg


def make_folds(df, cfg):
    cv_type = cfg["cv"]["type"]
    start_month = cfg["cv"].get("start_month", str(df.index[0].date()))
    fold_g = FoldGenerator(dt_index=df.index, mode=cv_type, start_month=start_month)

    if cv_type == "OddEven":
        return fold_g.make_two_month_folds()
    if cv_type == "Anchored":
        return fold_g.make_anchored_folds(
            embargo_hours=cfg["cv"].get("embargo_hours", 24),
            min_train_days=cfg["cv"].get("min_train_days", 30),
            test_freq=cfg["cv"].get("test_freq", "M"),
        )
    if cv_type == "Rolling":
        return fold_g.make_rolling_folds(
            cfg["cv"]["train_months"],
            cfg["cv"]["embargo_hours"],
            test_freq=cfg["cv"].get("test_freq", "M"),
        )
    raise ValueError(f"Unknown fold type: {cv_type}")


# ======================================================================
# Section C. 模型超參數 Suggest（LSTM / TemporalTransformer）
# ======================================================================
def suggest_model_hparams(trial: optuna.Trial, cfg: dict) -> dict:
    model_name = str(cfg["model"].get("name", "")).lower()

    if model_name in ["lstmhead", "lstm_se"]:
        cfg["model"]["hidden_size"]  = int(suggest_any(trial, "hidden_size", cfg["model"]["hidden_size"], type_hint="int"))
        cfg["model"]["n_layers"]     = int(suggest_any(trial, "n_layers",    cfg["model"]["n_layers"],    type_hint="int"))
        cfg["model"]["dropout"]      = float(suggest_any(trial, "dropout",   cfg["model"].get("dropout", 0.0), type_hint="float"))
        cfg["model"]["bidirectional"] = False

    elif model_name == "temporaltransformer":
        cfg["model"]["d_model"]  = int(suggest_any(trial, "d_model",  cfg["model"]["d_model"],  type_hint="int"))
        cfg["model"]["n_heads"]  = int(suggest_any(trial, "n_heads",  cfg["model"]["n_heads"],  type_hint="int"))
        if cfg["model"]["d_model"] % cfg["model"]["n_heads"] != 0:
            raise optuna.TrialPruned()

        cfg["model"]["n_layers"]    = int(suggest_any(trial, "n_layers",    cfg["model"].get("n_layers", 2), type_hint="int"))
        cfg["model"]["mlp_ratio"]   = float(suggest_any(trial, "mlp_ratio",  cfg["model"].get("mlp_ratio", 4.0), type_hint="float"))
        cfg["model"]["dropout"]     = float(suggest_any(trial, "dropout",    cfg["model"].get("dropout", 0.0), type_hint="float"))
        cfg["model"]["attn_dropout"]= float(suggest_any(trial, "attn_dropout", cfg["model"].get("attn_dropout", [0.0,0.1]), type_hint="float"))
        cfg["model"]["pooling"]     = suggest_any(trial, "pooling", cfg["model"].get("pooling", "attn"), type_hint="cat")

        # flags（bool 或 [True, False]）
        for key, default in [
            ("use_causal", True), ("use_alibi", True), ("use_conv_stem", True),
            ("use_input_norm", True), ("use_learned_pos", False)
        ]:
            cfg["model"][key] = bool(suggest_any(trial, key, cfg["model"].get(key, default), type_hint="bool"))

        # alibi_slope / droppath
        cfg["model"]["alibi_slope"] = float(suggest_any(trial, "alibi_slope", cfg["model"].get("alibi_slope", 0.05), type_hint="float"))
        cfg["model"]["droppath"]    = float(suggest_any(trial, "droppath",    cfg["model"].get("droppath", 0.0),   type_hint="float"))

        # 回歸任務強制使用 'attn' pooling（避免退化）
        if get_task_type(cfg) == "regression":
            cfg["model"]["pooling"] = "attn"

    # num_classes 填補
    task_type = get_task_type(cfg)
    cfg["model"]["num_classes"] = int(cfg["model"].get("num_classes", 2 if task_type=="classification" else 1))
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

    # ====== 這裡開始：把你原本的「訓練超參 + 回歸 loss 參數」整段替換掉 ======
    # ---- 訓練超參 ----
    cfg["train"]["lr"]            = float(suggest_any(trial, "lr", cfg["train"]["lr"], type_hint="float", log=True))
    cfg["train"]["weight_decay"]  = float(suggest_any(trial, "weight_decay", cfg["train"]["weight_decay"], type_hint="float", log=True))
    cfg["train"]["batch_size"]    = int(suggest_any(trial, "batch_size", cfg["train"]["batch_size"], type_hint="int"))
    cfg["train"]["grad_clip"]     = float(suggest_any(trial, "grad_clip", cfg["train"].get("grad_clip", 1.0), type_hint="float"))

    # 二分類 threshold（可 None / 單值 / 區間）
    thr_space = cfg["train"].get("threshold", None)
    cfg["train"]["threshold"] = suggest_any(trial, "threshold", thr_space, type_hint="float") if thr_space is not None else None

    # ---- 回歸的 loss 參數 ----
    if get_task_type(cfg) == "regression":
        cfg["loss"]["alpha"]     = float(suggest_any(trial, "loss.alpha",     cfg["loss"].get("alpha", 0.7),     type_hint="float"))
        cfg["loss"]["ema_decay"] = float(suggest_any(trial, "loss.ema_decay", cfg["loss"].get("ema_decay", 0.9), type_hint="float"))

        beta_space = cfg["loss"].get("beta", None)
        if beta_space is None:
            cfg["loss"]["beta"] = float(max(0.0, 1.0 - float(cfg["loss"]["alpha"])))
        else:
            cfg["loss"]["beta"] = float(suggest_any(trial, "loss.beta", beta_space, type_hint="float"))
    # ====== 這裡結束：替換區塊 ======

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