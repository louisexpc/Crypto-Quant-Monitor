# main_train.py
"""
Main entry for Optuna tuning with runtime-built features.

改動重點：
- 模組化分區（Load Cfg / Prepare DF / Build Study / Optimize / Postprocess
- Study 名稱包含 task / primary_metric / model 主要超參摘要
- 最佳結果輸出動態依 cfg.objective.primary_metric 與 direction 顯示
- 追加若干健壯性檢查與小優化（TF32、進度條、資料概況）
"""

from __future__ import annotations
from pathlib import Path

import optuna
import pandas as pd
import os
import multiprocessing as mp 
from train.pipeline.search.objective import objective
from train.core.context import set_seed
from train.core.config_loader import load_cfg
from train.data.dataloaders.base import load_precomputed_features, reindex_to_full_grid


__all__ = [
    "load_cfg",
    "prepare_dataframe",
    "build_study",
    "run_single",
    "run_multi",
]


# ======================================================================
# Section B. 準備資料（原始 OHLCV -> Indicators -> Features -> df[X]+label）
# ======================================================================
def prepare_dataframe(cfg: dict) -> pd.DataFrame:
    """
    預先讀取離線特徵檔，只取時間索引用於後續折疊生成。
    """
    feat_df = load_precomputed_features(path=cfg["data"]["path"])
    freq = (cfg.get("data", {}) or {}).get("freq")
    if freq and not feat_df.empty:
        feat_df = reindex_to_full_grid(feat_df, str(freq))

    # 沿用舊行為：只取 index 骨架給 make_folds 用
    df = pd.DataFrame(index=feat_df.index)
    df = df[~df.index.duplicated(keep="last")]
    return df

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
def run_single(cfg_path: str, *, worker_tag: str | None = None, cfg: dict | None = None):
    # 1) 載入設定 / 設定隨機種子 / 啟動 CUDA 選項
    cfg = cfg or load_cfg(cfg_path)
    set_seed(int(cfg.get("seed", 42)))

    # 2) 建輸出資料夾
    run_dir = Path("runs") / cfg.get("project_name", "exp")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 將 worker_tag 寫入環境變數（objective.py 會用來決定 trial 子資料夾）
    if worker_tag:
        os.environ["WORKER_TAG"] = str(worker_tag)

    # 3) 準備資料
    df = prepare_dataframe(cfg)

    # 4) 建立 Study（parallel=True 時 sampler 會開 constant_liar）
    study = build_study(cfg, run_dir, parallel=bool(worker_tag))

    # 5) 搜尋
    n_trials = int(cfg["search"]["n_trials"])
    time_hour = int(cfg["search"]["timeout"])  # 小時
    study.optimize(
        lambda t: objective(t, cfg, df, run_dir),
        n_trials=n_trials,
        timeout=time_hour * 60 * 60,
        show_progress_bar=not bool(worker_tag)  # worker 不顯示進度條，避免互相干擾
    )

    print("Best hyperparameters:", study.best_trial.params)
    print(f"Best `{cfg['objective']['primary_metric']}` ({cfg['objective']['direction']}): {study.best_value:.6g}")

def _worker_entry(cfg_path: str, gpu_id: int):
    # 每個進程只看到自己那張卡，且標上 worker_tag
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    worker_tag = f"gpu{gpu_id}"
    # 建議用 spawn（WSL/CUDA 友好）
    run_single(cfg_path, worker_tag=worker_tag)

def run_multi(cfg_path: str, *, cfg: dict | None = None):
    cfg = cfg or load_cfg(cfg_path)
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
    cfg_path = "train/config.yaml"
    cfg = load_cfg(cfg_path)
    mg = cfg.get("search", {}).get("multi_gpu", {}) or {}
    if mg.get("enabled", False):
        run_multi(cfg_path, cfg=cfg)
    else:
        run_single(cfg_path, cfg=cfg)
