# train/core/orchestrator.py
from __future__ import annotations
from pathlib import Path
from importlib import import_module
from typing import Callable
from .config_loader import load_cfg
from .context import set_seed

def run(cfg_path: str | Path, entry: Callable[[str], None]) -> None:
    """
    1. 說明: 薄 orchestrator：載入設定、設 seed，轉呼舊入口
    2. inputs:
       - cfg_path (str|Path): YAML 路徑
       - entry (Callable): 舊的主入口，例如 train.main_train.run_single
    3. return: None
    """
    cfg = load_cfg(cfg_path)
    set_seed(int(cfg.get("seed", 42)), deterministic=cfg.get("deterministic", True))
    entry(str(cfg_path))

if __name__ == "__main__":
    # 簡易 CLI：python -m train.core.orchestrator train/config.yaml
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "train/config.yaml"
    mm = import_module("train.main_train")
    run(cfg_path, getattr(mm, "run_single"))
