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
from utils.dataloader import load_pt_cache
from export_metrices import dump_best_yaml


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
    use_pt = bool(cfg["data"].get("use_pt_cache", False))
    if use_pt:
        X_cpu, y_cpu, ts_cpu, cols = load_pt_cache(cfg["data"]["pt_path"])
        assert ts_cpu is not None, "你的 .pt 檔缺少 'ts'（時間索引）"
        assert cols is not None, "你的 .pt 檔缺少 'cols'（特徵名）"

        df = pd.DataFrame(X_cpu.numpy(), index=ts_cpu, columns=cols)
        df["label"] = y_cpu.numpy()
        df = df.sort_index()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        pt_bundle = {
            "X": X_cpu,
            "Y": y_cpu,
            "TS": ts_cpu,
            "FEAT_COLS": cols
        }
        return df, pt_bundle

    else:
        df = pd.read_csv(cfg["data"]["path"], parse_dates=[cfg["data"]["index_col"]]).set_index(cfg["data"]["index_col"])
        if "label" not in df.columns:
            h = int(cfg["label"]["horizon"])
            band_bps = cfg["label"]["flat_band_bps"]
            if isinstance(band_bps, list):
                band_bps = int(np.mean(band_bps))
            band = float(band_bps) / 10000.0
            ret = np.log(df["close"].shift(-h) / df["close"])
            label = pd.Series(np.where(ret >= band, 2, np.where(ret <= -band, 0, 1)), index=df.index, name="label")
            df = pd.concat([df, label], axis=1)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df, None


def run(cfg_path: str):
    cfg = load_cfg(cfg_path)
    set_seed(int(cfg.get("seed", 42)))
    torch.set_float32_matmul_precision("high")

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
    print("Best val macro-prec:", study.best_value)

    dump_best_yaml(study, cfg, run_dir)


if __name__ == "__main__":
    run("train/config.yaml")
