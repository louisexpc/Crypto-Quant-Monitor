# train/core/context.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os, random, numpy as np
import torch, warnings

@dataclass(frozen=True)
class RunContext:
    """
    1. 說明: 保存一次執行所需的不可變上下文（cfg、路徑、seed）
    2. inputs:
       - cfg: 設定 dict
       - run_dir: 輸出根目錄
       - seed: 隨機種子
    3. return:
       - RunContext: 不可變物件
    """
    cfg: Dict[str, Any]
    run_dir: Path
    seed: int

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    1. 說明: 一次性設定 Python/numpy/torch 的亂數
    2. inputs:
       - seed: 整數種子
       - deterministic: 是否啟用 deterministic 模式
    3. return: None
    """
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    # benchmark 與 deterministic 互斥；若採 deterministic，仍維持 False
    torch.backends.cudnn.benchmark = False if deterministic else True

    # CUDA / CUDNN 加速設定
    torch.backends.cuda.matmul.fp32_precision = 'tf32'   # 'ieee' = 關閉 TF32
    torch.backends.cudnn.conv.fp32_precision  = 'tf32'

    try:
        # 新版 PyTorch 針對 SDPA 的控制；若不可用則忽略
        from torch.nn.attention import sdpa_kernel  # type: ignore
        _ = sdpa_kernel  # 僅確認 import 成功
    except Exception:
        pass

    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated.*",
        category=UserWarning,
        module="pandas_ta"
    )
