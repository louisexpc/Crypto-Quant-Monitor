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

import optuna
import pandas as pd
import yaml
import os
import multiprocessing as mp 
from train.objective.objective import objective
from train.train_utils.compute_export_metrices import dump_best_yaml
from train.train_utils.init_train import setup_cuda_acceleration, set_seed
from train.data.feature_store import FeatureStore  


__all__ = [
    "load_cfg",
    "prepare_dataframe",
    "build_study",
    "run_single",
    "run_multi",
]

# ======================================================================
# Section A. 基本工具
# ======================================================================
def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ======================================================================
# Section B. 準備資料（原始 OHLCV -> Indicators -> Features -> df[X]+label）
# ======================================================================
# def prepare_dataframe(cfg: dict) -> tuple[pd.DataFrame, dict | None]:
#     """
#     Precomputed-only: use features.precomputed.path to provide the time index
#     for fold generation. No runtime feature computation.
#     """
#     pre_path = cfg["data"]["path"]
#     if not pre_path:
#         raise ValueError("請在 config.features.precomputed.path 指定預先計算的特徵檔 (.csv 或 .parquet)")
#     p = str(pre_path)
#     if p.endswith(".csv"):
#         dfp = pd.read_csv(p)
#     elif p.endswith(".parquet"):
#         dfp = pd.read_parquet(p)
#     else:
#         raise ValueError("features.precomputed.path 只支援 .csv 或 .parquet")

#     # infer index from datetime/timestamp or existing DatetimeIndex
#     if isinstance(dfp.index, pd.DatetimeIndex):
#         idx = dfp.index
#     elif "datetime" in dfp.columns:
#         idx = pd.to_datetime(dfp["datetime"], errors="coerce", utc=True)
#     elif "timestamp" in dfp.columns:
#         ts = pd.to_numeric(dfp["timestamp"], errors="coerce").astype("Int64")
#         unit = "ms" if (ts.dropna().iloc[0] if len(ts.dropna()) else 0) > 1_000_000_000_000 else "s"
#         idx = pd.to_datetime(ts, unit=unit, utc=True)
#     else:
#         raise ValueError("預算特徵檔需包含 'datetime' 或 'timestamp' 欄位，或已是 DatetimeIndex")

#     df = pd.DataFrame(index=pd.DatetimeIndex(idx).sort_values())
#     df = df[~df.index.duplicated(keep="last")]
#     return df, None

def prepare_dataframe(cfg: dict) -> tuple[pd.DataFrame, dict | None]:
    """
    Precomputed-only: use data.path to provide the time index for fold generation.
    另外建立 FeatureStore，放到 pt_bundle 交給 objective 使用。
    """
    # 建立 FeatureStore（會完成讀檔與 UTC index）
    fs = FeatureStore.from_cfg(cfg, compute_time_labels=True)

    # 沿用舊行為：只取 index 骨架給 make_folds 用
    df = pd.DataFrame(index=fs.get_frame().index)
    df = df[~df.index.duplicated(keep="last")]
    return df, {"feature_store": fs}

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
    # trials = study.get_trials(deepcopy=False)
    # from collections import Counter
    # from optuna.trial import TrialState
    # sc = Counter([getattr(t, "state", None) for t in trials])
    # sc_fmt = {str(k).split('.')[-1]: int(v) for k, v in sc.items()}
    # print(f"[Optuna] Trial states: {sc_fmt}")

    # has_complete = any(t.state == TrialState.COMPLETE for t in trials)
    # if not has_complete:
    #     print("[Optuna] No completed trials (all pruned/failed). Skip best summary/export.")
    #     return

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
    cfg_path = "train/config.yaml"
    cfg = load_cfg(cfg_path)
    mg = cfg.get("search", {}).get("multi_gpu", {}) or {}
    if mg.get("enabled", False):
        run_multi(cfg_path)
    else:
        run_single(cfg_path)
