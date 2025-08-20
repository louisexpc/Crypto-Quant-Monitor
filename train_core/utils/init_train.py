# init_train.py

import numpy as np
import torch
import random
import pandas as pd


# ---------------------------
# 工具：設定隨機種子
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False

def setup_cuda_acceleration():
    import torch

    # MatMul：建議 'high'（允許 TF32 / 更快），或 'medium'
    torch.set_float32_matmul_precision("high")
    # 直接指定後端行為（PyTorch 2.5+ 提供的新 API）
    torch.backends.cuda.matmul.fp32_precision = "tf32"    # 可選: "ieee" 關閉 TF32
    torch.backends.cudnn.conv.fp32_precision  = "tf32"    # cuDNN 卷積用 TF32

    torch.backends.cudnn.benchmark = True

    # SDP 設定（可用新介面；舊 enable_* 也可，建議用新）
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=True,
                                        enable_math=False,
                                        enable_mem_efficient=True)
    except Exception:
        pass








