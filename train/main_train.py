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
import argparse
from pathlib import Path
from typing import Any, Dict
import random
import warnings
import optuna
import pandas as pd
import numpy as np
import torch
import yaml
from train.pipeline.search.objective import objective
from train.data.dataloaders.base import (
    load_precomputed_features,
    reindex_to_full_grid,
    resolve_timezone_name,
)


__all__ = [
    "load_cfg",
    "prepare_dataframe",
    "build_study",
    "run",
]


def load_cfg(path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Configuration file path.

    Returns:
        Parsed configuration dictionary.
    """
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducible training.

    Args:
        seed: Random seed value.
        deterministic: Whether to enable deterministic CUDA behavior.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        pass
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated.*",
        category=UserWarning,
        module="pandas_ta",
    )


# ======================================================================
# Section B. 準備資料（原始 OHLCV -> Indicators -> Features -> df[X]+label）
# ======================================================================
def prepare_dataframe(cfg: dict) -> pd.DataFrame:
    """
    預先讀取離線特徵檔，只取時間索引用於後續折疊生成。
    """
    data_cfg = (cfg.get("data", {}) or {})
    feat_path = data_cfg.get("feat_path")
    if not feat_path:
        raise KeyError("cfg.data.feat_path is required.")
    data_tz = resolve_timezone_name(data_cfg.get("time_zone"), default="Asia/Taipei")

    feat_df = load_precomputed_features(path=feat_path, target_tz=data_tz)
    freq = data_cfg.get("freq")
    if freq and not feat_df.empty:
        feat_df = reindex_to_full_grid(feat_df, str(freq))
    df = pd.DataFrame(index=feat_df.index)
    df = df[~df.index.duplicated(keep="last")]
    return df

# ======================================================================
# Section C. 建立 Study（Optuna）
# ======================================================================
def build_study(cfg: dict, run_dir: Path, *, parallel: bool = False) -> optuna.Study:
    study_name = cfg["project_name"]

    # --- SQLite URI with timeout for concurrency ---
    sqlite_timeout = int(cfg.get("search", {}).get("sqlite_timeout_sec", 120))
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
def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(cfg)

    def ensure_branch(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
        node = parent.get(key)
        if not isinstance(node, dict):
            node = {}
            parent[key] = node
        return node

    if args.project_name:
        cfg["project_name"] = args.project_name

    if args.feat_path:
        ensure_branch(cfg, "data")["feat_path"] = args.feat_path

    if args.ohlcv_path:
        ensure_branch(cfg, "data")["ohlcv_fng_path"] = args.ohlcv_path

    if args.micro_path:
        data_branch = ensure_branch(cfg, "data")
        micro_branch = ensure_branch(data_branch, "micro")
        micro_branch["path"] = args.micro_path

    if args.tbm_csv_path or args.tbm_keep_sides:
        label_branch = ensure_branch(cfg, "label")
        if args.tbm_csv_path:
            label_branch["tbm_csv_path"] = args.tbm_csv_path
        if args.tbm_keep_sides:
            label_branch["keep_sides"] = args.tbm_keep_sides

    if args.n_trials is not None:
        ensure_branch(cfg, "search")["n_trials"] = int(args.n_trials)

    return cfg


def run(cfg_path: str, *, cfg: dict | None = None):
    # 1) 載入設定 / 設定隨機種子 / 啟動 CUDA 選項
    cfg = cfg or load_cfg(cfg_path)
    set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", True)))

    # 2) 建輸出資料夾
    run_dir = Path("runs") / cfg.get("project_name", "exp")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3) 準備資料
    df = prepare_dataframe(cfg)

    # 4) 建立 Study（parallel=True 時 sampler 會開 constant_liar）
    study = build_study(cfg, run_dir, parallel=False)

    # 5) 搜尋
    n_trials = int(cfg["search"]["n_trials"])
    time_hour = int(cfg["search"]["timeout"])  # 小時
    study.optimize(
        lambda t: objective(t, cfg, df, run_dir),
        n_trials=n_trials,
        timeout=time_hour * 60 * 60,
        show_progress_bar=True
    )

    print("Best hyperparameters:", study.best_trial.params)
    print(f"Best `{cfg['objective']['primary_metric']}` ({cfg['objective']['direction']}): {study.best_value:.6g}")

# ======================================================================
# Section E. 入口
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna TBM training entry")
    parser.add_argument("--config", default="train/config.yaml", help="Path to config YAML.")
    parser.add_argument("--project-name", help="Override cfg.project_name.")
    parser.add_argument("--feat-path", help="Override cfg.data.feat_path.")
    parser.add_argument("--ohlcv-path", help="Override cfg.data.ohlcv_fng_path.")
    parser.add_argument("--micro-path", help="Override cfg.data.micro.path.")
    parser.add_argument("--tbm-csv-path", help="Override cfg.label.tbm_csv_path.")
    parser.add_argument("--tbm-keep-sides", choices=["short", "long", "both"], help="Override cfg.label.keep_sides.")
    parser.add_argument("--n-trials", type=int, help="Override cfg.search.n_trials.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    cfg = _apply_cli_overrides(cfg, args)
    run(args.config, cfg=cfg)
