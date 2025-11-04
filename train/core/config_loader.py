# train/core/config_loader.py
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict

def load_cfg(path: str | Path) -> Dict[str, Any]:
    """
    1. 說明: 讀取 YAML 設定並回傳 dict（後續可接 schema 驗證）
    2. inputs:
       - path: 設定檔路徑
    3. return:
       - dict: 解析後的設定
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
