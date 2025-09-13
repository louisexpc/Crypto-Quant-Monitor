# time_dataset.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional, List, Dict

class SeqDataset(Dataset):
    def __init__(
        self,
        X_df,
        y_s,
        seq_len: int,
        scaler=None,
        device: str = "cuda",
        label_dtype: Literal["auto", "float", "long"] = "auto",
        stride: int = 1,                 # ★ 新增
        anchor: int = 0,                 # ★ 新增：0..stride-1，控制起始對齊
    ):
        X_df = X_df.astype(np.float32, copy=False)

        if scaler is None:
            X = X_df.values
        elif hasattr(scaler, "transform"):
            X = scaler.transform(X_df.values).astype(np.float32, copy=False)
        elif hasattr(scaler, "transform_df"):
            X = scaler.transform_df(X_df).values.astype(np.float32, copy=False)
        else:
            raise TypeError("Unsupported scaler: expected .transform(...) or .transform_df(...)")

        # ---- y ----
        if label_dtype == "auto":
            is_float = np.issubdtype(y_s.values.dtype, np.floating)
            y_np = y_s.values.astype(np.float32 if is_float else np.int64, copy=False)
            torch_y_dtype = torch.float32 if is_float else torch.long
        elif label_dtype == "float":
            y_np = y_s.values.astype(np.float32, copy=False)
            torch_y_dtype = torch.float32
        else:
            y_np = y_s.values.astype(np.int64, copy=False)
            torch_y_dtype = torch.long

        # ---- sliding windows with stride ----
        L = int(seq_len)
        N, M = len(X), len(y_np)
        stride = max(1, int(stride))
        anchor = int(anchor) % stride

        start = (L - 1) + anchor
        stop = min(N, M)
        if start >= stop:
            # 沒有任何可用樣本時，回退到無 anchor
            start = L - 1

        idx = np.arange(start=start, stop=stop, step=stride, dtype=int)

        # [N, T, F] / [N]
        X_seqs = np.stack([X[j - L + 1: j + 1] for j in idx]) if len(idx) else np.empty((0, L, X.shape[1]), np.float32)
        y_vals = y_np[idx] if len(idx) else np.empty((0,), y_np.dtype)

        self.X = torch.tensor(X_seqs, dtype=torch.float32, device=device)
        self.y = torch.tensor(y_vals, dtype=torch_y_dtype, device=device)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

