# main_train.py
"""
Main entry for Optuna tuning with runtime-built features.

改動重點：
- 模組化分區（Load Cfg / Prepare DF / Build Study / Optimize / Postprocess）
- dump_best_yaml 改從 config_export 匯入（與 trainer 對齊）
- Study 名稱包含 task / primary_metric / model 主要超參摘要
- 最佳結果輸出動態依 cfg.objective.primary_metric 與 direction 顯示
- 追加若干健壯性檢查與小優化（TF32、進度條、資料概況）
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

# ---- Objective ----
from objective_runtime import objective

# ---- Exports / Reporting（與 trainer.py 同源）----
from compute_export_metrices import dump_best_yaml

# ---- Runtime features ----
from utils.init_train import setup_cuda_acceleration, set_seed
from utils.indicators_runtime import Indicators, PAPER_TOP8_PLAN
from utils.build_features import build_features_and_label_runtime


# ======================================================================
# Section A. 基本工具
# ======================================================================
def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ======================================================================
# Section B. 準備資料（原始 OHLCV -> Indicators -> Features -> df[X]+label）
# ======================================================================
def prepare_dataframe(cfg: dict) -> tuple[pd.DataFrame, dict | None]:
    path = cfg["data"]["path"]
    index_col = cfg["data"]["index_col"]      # "timestamp" or "datetime"
    freq = cfg["data"].get("freq", None)

    task_type = (cfg.get("task", {}) or {}).get("type", "classification").lower()
    true_thr  = float(cfg.get("regression_to_class", {}).get("true_threshold", 0.0))

    # 讀原始 OHLCV
    if path.endswith(".csv"):
        df_raw = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df_raw = pd.read_parquet(path)
    else:
        raise ValueError("只支援 .csv 或 .parquet 檔案")

    # Indicators（照你原有流程）
    ind = Indicators(
        df_raw,
        cache_dir=cfg["features"].get("cache_dir", "cache_features"),
        freq_check=freq,
        prefer_time_col=index_col,
    )
    plan = cfg["features"].get("plan", PAPER_TOP8_PLAN)
    feat_df = ind.compute(plan)
    feat_path = ind.cache_path_for(plan)

    # ★ 關鍵：把 task_type / cls_threshold 傳入，回歸→連續 target、分類→二值 label
    X_df, y_s = build_features_and_label_runtime(
        df_base=ind.df,
        feat_parquet_path=feat_path,
        feat_df=feat_df,
        horizon=int(cfg["label"].get("horizon", 1)),
        ret_kind=cfg["label"].get("ret_kind", "logret"),
        task_type=task_type,
        cls_threshold=true_thr,
    )

    # 小檢查與資訊
    if task_type == "regression":
        print(f"[Data] regression target stats: mean={y_s.mean():.4e}, std={y_s.std():.4e}, "
              f"min={y_s.min():.4e}, max={y_s.max():.4e}")
        target_col = "target"  # 由 build_features_and_label_runtime 命名
    else:
        print(f"[Data] label unique={y_s.nunique()}, dist={y_s.value_counts(normalize=True).round(4).to_dict()}")
        target_col = "label"

    # 組 df：特徵 + 目標
    df = pd.concat([X_df, y_s], axis=1)
    assert target_col in df.columns, f"建表失敗：df 缺少 '{target_col}' 欄位。"

    pt_bundle = {"feat_parquet": str(feat_path), "target_col": target_col, "task_type": task_type}
    return df, pt_bundle


# ======================================================================
# Section C. 建立 Study（Optuna）
# ======================================================================
def build_study(cfg: dict, run_dir: Path) -> optuna.Study:
    study_name = cfg.get("project_name", "study")
    task = (cfg.get("task", {}) or {}).get("type", "classification")
    primary = (cfg.get("objective", {}) or {}).get("primary_metric", "macro_f1")
    direction = (cfg.get("objective", {}) or {}).get("direction", "maximize")

    # 模型摘要（讓 DB 名稱更具體）
    def _sig(lst):
        if isinstance(lst, list):
            return "-".join(map(str, lst))
        return str(lst)

    m = cfg["model"]
    if m.get("name", "").lower() == "temporaltransformer":
        study_suffix = f"d{_sig(m.get('d_model', []))}_h{_sig(m.get('n_heads', []))}_L{_sig(m.get('n_layers', []))}"
    else:
        study_suffix = f"hs{_sig(m.get('hidden_size', []))}_nl{_sig(m.get('n_layers', []))}"

    study_name = f"{study_name}__{task}_{primary}__{study_suffix}"

    db_uri = f"sqlite:///{(run_dir / 'study.db').as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        storage=db_uri,
        load_if_exists=True,
        direction=direction,  # "maximize" / "minimize" 與 objective_runtime 保持一致
        sampler=optuna.samplers.TPESampler(
            seed=cfg["search"].get("seed", 2025),
            multivariate=True,          # 通常更穩
            group=True
        ),
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=cfg["search"].get("pruner_warmup_folds", 2)  # 以「fold」作為 step
        ),
    )
    return study


# ======================================================================
# Section D. 執行訓練與搜尋
# ======================================================================
def run(cfg_path: str):
    # 1) 載入設定 / 設定隨機種子 / 啟動 CUDA 選項
    cfg = load_cfg(cfg_path)
    set_seed(int(cfg.get("seed", 42)))
    setup_cuda_acceleration()

    # 2) 建輸出資料夾
    run_dir = Path("runs") / cfg.get("project_name", "exp")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3) 準備資料
    df, pt_bundle = prepare_dataframe(cfg)

    # 4) 建立 Study
    study = build_study(cfg, run_dir)

    # 5) 搜尋（以 folds 為 step 報告分數；objective_runtime 會依 cfg.objective 自動處理 direction）
    n_trials = int(cfg["search"]["n_trials"])
    time_hour = int(cfg["search"]["timeout"])  # 小時
    study.optimize(
        lambda t: objective(t, cfg, df, run_dir, pt_bundle),
        n_trials=n_trials,
        timeout=time_hour * 60 * 60,
        show_progress_bar=True
    )

    # 6) 結果輸出
    print("Best hyperparameters:", study.best_trial.params)
    print(f"Best `{cfg['objective']['primary_metric']}` ({cfg['objective']['direction']}): {study.best_value:.6g}")

    # 匯出最佳 trial 的 YAML（包含實際 frozen config 與 selected_features）
    dump_best_yaml(study, cfg, run_dir)


# ======================================================================
# Section E. 入口
# ======================================================================
if __name__ == "__main__":
    # 例如：train/config_runtime.yaml
    run("train/config_runtime.yaml")
