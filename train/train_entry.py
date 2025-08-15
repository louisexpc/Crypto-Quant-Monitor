# train_entry.py

import os
import copy
import yaml
import torch
import random
import numpy as np
import pandas as pd
import optuna
from pathlib import Path

from objective import objective
# from train.utils.dataloader_old import load_pt_cache
from export_metrices import dump_best_yaml

from utils.cuda_utils import setup_cuda_acceleration


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_dataframe(cfg: dict) -> tuple[pd.DataFrame, dict | None]:
    path = cfg["data"]["path"]
    index_col = cfg["data"]["index_col"]

    # ===  根據副檔名選擇讀取方式 ===
    if path.endswith(".csv"):
        df = pd.read_csv(path, parse_dates=[index_col])
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
        if index_col not in df.columns and df.index.name != index_col:
            raise ValueError(f"Parquet 檔缺少 index_col: {index_col}")
        if df.index.name != index_col:
            df = df.set_index(index_col)
        df.index = pd.DatetimeIndex(df.index)
    else:
        raise ValueError("只支援 .csv 或 .parquet 檔案")

        # # === ✅ 自動加上 label（若缺） ===
        # if "label" not in df.columns:
        #     h = int(cfg["label"]["horizon"])
        #     band_bps = cfg["label"]["flat_band_bps"]
        #     if isinstance(band_bps, list):
        #         band_bps = int(np.mean(band_bps))
        #     band = float(band_bps) / 10000.0
        #     ret = np.log(df["close"].shift(-h) / df["close"])
        #     label = pd.Series(np.where(ret >= band, 2, np.where(ret <= -band, 0, 1)), index=df.index, name="label")
        #     df = pd.concat([df, label], axis=1)

        # df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, None


def run(cfg_path: str):
    cfg = load_cfg(cfg_path)
    set_seed(int(cfg.get("seed", 42)))
    setup_cuda_acceleration()

    run_dir = Path("runs") / cfg.get("project_name", "exp")
    run_dir.mkdir(parents=True, exist_ok=True)

    df, pt_bundle = prepare_dataframe(cfg)

    def _sig(lst):
        if isinstance(lst, list):
            return "-".join(map(str, lst))
        return str(lst)

    study_name = cfg.get("project_name", "study")
    study_suffix = f"hs{_sig(cfg['model'].get('hidden_size', []))}_nl{_sig(cfg['model'].get('n_layers', []))}"
    study_name = f"{study_name}_{study_suffix}"

    db_uri = f"sqlite:///{(run_dir / 'study.db').as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        storage=db_uri,
        load_if_exists=True,
        direction=cfg["objective"]["direction"],
        sampler=optuna.samplers.TPESampler(seed=cfg["search"].get("seed", 2025)),
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=cfg["search"].get("pruner_warmup_folds", 2)
        ),
    )

    n_trials = int(cfg["search"]["n_trials"])
    time_hour = int(cfg["search"]["timeout"])

    study.optimize(lambda t: objective(t, cfg, df, run_dir, pt_bundle),
                   n_trials=n_trials,
                   timeout=time_hour * 60 * 60,
                   show_progress_bar=True)

    print("Best hyperparameters:", study.best_trial.params)
    print("Best val macro-f_05:", study.best_value)

    dump_best_yaml(study, cfg, run_dir)


if __name__ == "__main__":
    run("train/config.yaml")
