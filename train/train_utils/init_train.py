# init_train.py

import numpy as np
import torch
import random
import pandas as pd


# ---------------------------
# 工具：設定隨機種子
# ---------------------------
def set_seed(seed: int, deterministic = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic


def setup_cuda_acceleration():
    import torch, warnings
    # ---- TF32（新 API）----
    torch.backends.cuda.matmul.fp32_precision = 'tf32'   # 'ieee' = 關閉 TF32
    torch.backends.cudnn.conv.fp32_precision  = 'tf32'
    torch.backends.cudnn.benchmark = True

    # ---- SDPA（新 API）----
    # 以前: torch.backends.cuda.sdp_kernel(...)
    # 現在改成：
    try:
        from torch.nn.attention import sdpa_kernel
        # 建議在訓練主程式用: with sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True): ...
        # 若你想全域包起來，可在外層呼叫這個 context。這裡只示範可用性檢查。
    except Exception as e:
        pass  # 舊卡或舊版 torch 會沒有，忽略

    # ---- 靜音第三方 warning（可選）----
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated.*",
        category=UserWarning,
        module="pandas_ta"
    )





