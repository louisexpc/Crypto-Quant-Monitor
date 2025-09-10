# # main_train.py
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
import os
import multiprocessing as mp 

# ---- Objective ----
from objective.objective import objective

# ---- Exports / Reporting（與 trainer.py 同源）----
from train_utils.compute_export_metrices import dump_best_yaml

# ---- Runtime features ----
from train_utils.init_train import setup_cuda_acceleration, set_seed
from build_feature_loader.indicators import IndicatorLibrary, FeatureComputer
from build_feature_loader.build_features import build_features_and_label


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

    task_type = str(cfg["task"]["type"]).lower()

    # 讀原始 OHLCV
    if path.endswith(".csv"):
        df_raw = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df_raw = pd.read_parquet(path)
    else:
        raise ValueError("只支援 .csv 或 .parquet 檔案")

    # 1) 規格化 OHLCV
    lib = IndicatorLibrary(
        df_raw,
        freq_check=cfg["data"]["freq"],        # 例如 "1H"
        prefer_time_col=index_col,             # "timestamp" 或 "datetime"
    )
    # 2) 建 特徵計算器（內建 parquet + manifest 快取）
    #    多 GPU 併行時，每個 worker 用不同的 cache 子資料夾，避免寫入競態
    cache_dir = cfg["features"]["cache_dir"]
    worker_tag = os.environ.get("WORKER_TAG", "").strip()
    if worker_tag:
        cache_dir = str((Path(cache_dir) / worker_tag).as_posix())
    fc = FeatureComputer(lib, cache_dir=cache_dir)

    # 3) 計算或查詢特徵
    plan = cfg["features"]["plan"]

    # 事件模式：避免預先重算全量特徵，改走 manifest 欄名，並以原始 OHLCV 的 index 作為時間基準
    label_mode = str(cfg.get("label", {}).get("mode", "")).lower()
    if label_mode == "event_tbm":
        feat_cols = fc.columns_for_plan(plan, cfg)  # 若 manifest 缺少，會觸發一次 compute 以生成
        # 本階段只需時間索引供 fold 切分；不需要 y 或特徵值
        df = lib.df.copy()
        target_col = "label"
        pt_bundle = {"target_col": target_col, "task_type": task_type, "feat_cols": feat_cols}
        return df, pt_bundle
    else:
        feat_df = fc.compute(plan, cfg)    # -> 已 shift(1) 防洩漏、已快取

        #   回歸→連續 target、分類→二值 label
        X_df, y_s = build_features_and_label(
            df_base=lib.df,
            feat_parquet_path=None,
            feat_df=feat_df,
            cfg=cfg)
        

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

    # pt_bundle = {"feat_parquet": str(feat_path), "target_col": target_col, "task_type": task_type}
    pt_bundle = {
        "target_col": target_col,
        "task_type": task_type,
        "feat_cols": feat_df.columns.tolist(),    # ← 傳出去
    }
    return df, pt_bundle


# ======================================================================
# Section C. 建立 Study（Optuna）
# ======================================================================
def build_study(cfg: dict, run_dir: Path, *, parallel: bool = False) -> optuna.Study:
    study_name = cfg["project_name"]

    # --- SQLite URI with timeout for concurrency ---
    mg = cfg.get("search", {}).get("multi_gpu", {}) or {}
    sqlite_timeout = int(mg.get("sqlite_timeout_sec", 120))
    db_uri = f"sqlite:///{(run_dir / 'study.db').as_posix()}?timeout={sqlite_timeout}"

    # --- Sampler: 平行建議 constant_liar=True ---
    sampler = optuna.samplers.TPESampler(
        seed=cfg["search"]["seed"],
        multivariate=True,
        group=True,
        constant_liar=bool(parallel),  # 平行才開
    )

    # Respect enable_prune flag
    use_pruner = bool(cfg.get("objective", {}).get("enable_prune", True))
    pruner = optuna.pruners.MedianPruner(
        n_warmup_steps=cfg["search"]["pruner_warmup_folds"]
    ) if use_pruner else None

    study = optuna.create_study(
        study_name=study_name,
        storage=db_uri,
        load_if_exists=True,
        direction=cfg["objective"]["direction"],  # "maximize" / "minimize" 與 objective_runtime 保持一致
        sampler=sampler,
        pruner=pruner,
    )
    return study


# ======================================================================
# Section D. 執行訓練與搜尋
# ======================================================================
def run_single(cfg_path: str, *, worker_tag: str | None = None):
    # 1) 載入設定 / 設定隨機種子 / 啟動 CUDA 選項
    cfg = load_cfg(cfg_path)
    set_seed(int(cfg.get("seed", 42)))
    setup_cuda_acceleration()

    # 2) 建輸出資料夾
    run_dir = Path("runs") / cfg.get("project_name", "exp")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 將 worker_tag 寫入環境變數（objective.py 會用來決定 trial 子資料夾）
    if worker_tag:
        os.environ["WORKER_TAG"] = str(worker_tag)

    # 3) 準備資料
    df, pt_bundle = prepare_dataframe(cfg)

    # 4) 建立 Study（parallel=True 時 sampler 會開 constant_liar）
    study = build_study(cfg, run_dir, parallel=bool(worker_tag))

    # 5) 搜尋
    n_trials = int(cfg["search"]["n_trials"])
    time_hour = int(cfg["search"]["timeout"])  # 小時
    study.optimize(
        lambda t: objective(t, cfg, df, run_dir, pt_bundle),
        n_trials=n_trials,
        timeout=time_hour * 60 * 60,
        show_progress_bar=not bool(worker_tag)  # worker 不顯示進度條，避免互相干擾
    )

    # 6) 結果輸出（容錯：若全部被 PRUNED/FAILED，避免崩潰）
    trials = study.get_trials(deepcopy=False)
    from collections import Counter
    from optuna.trial import TrialState
    sc = Counter([getattr(t, "state", None) for t in trials])
    sc_fmt = {str(k).split('.')[-1]: int(v) for k, v in sc.items()}
    print(f"[Optuna] Trial states: {sc_fmt}")

    has_complete = any(t.state == TrialState.COMPLETE for t in trials)
    if not has_complete:
        print("[Optuna] No completed trials (all pruned/failed). Skip best summary/export.")
        return

    print("Best hyperparameters:", study.best_trial.params)
    print(f"Best `{cfg['objective']['primary_metric']}` ({cfg['objective']['direction']}): {study.best_value:.6g}")
    dump_best_yaml(study, cfg, run_dir)


def _worker_entry(cfg_path: str, gpu_id: int):
    # 每個進程只看到自己那張卡，且標上 worker_tag
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    worker_tag = f"gpu{gpu_id}"
    # 建議用 spawn（WSL/CUDA 友好）
    run_single(cfg_path, worker_tag=worker_tag)

def run_multi(cfg_path: str):
    cfg = load_cfg(cfg_path)
    mg = cfg.get("search", {}).get("multi_gpu", {}) or {}
    gpu_ids = list(mg.get("gpu_ids", [0, 1]))
    assert len(gpu_ids) >= 2, "multi_gpu.enabled=true 但 gpu_ids 少於 2"

    mp.set_start_method("spawn", force=True)
    procs = []
    for gid in gpu_ids:
        p = mp.Process(target=_worker_entry, args=(cfg_path, int(gid)), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


# ======================================================================
# Section E. 入口
# ======================================================================
if __name__ == "__main__":
    import xgboost as xgb
    # print("XGB=", xgb.__version__)
    cfg_path = "train/config.yaml"

    cfg = load_cfg(cfg_path)
    mg = cfg.get("search", {}).get("multi_gpu", {}) or {}
    if mg.get("enabled", False):
        run_multi(cfg_path)
    else:
        run_single(cfg_path)
