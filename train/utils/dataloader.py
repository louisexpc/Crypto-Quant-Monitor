
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from typing import Literal, Optional

# 小工具：從 cfg 判斷任務
def _task_type_from_cfg(cfg: dict) -> str:
    if "task" in cfg and "type" in cfg["task"]:
        return str(cfg["task"]["type"]).lower()
    if "target" in cfg and "type" in cfg["target"]:
        return str(cfg["target"]["type"]).lower()
    return "classification" if int(cfg["model"].get("num_classes", 1)) >= 2 else "regression"

def split_fold_to_indices(df: pd.DataFrame, fold: dict, cfg: dict):
    train_val_mask = fold["train_val_mask"]
    test_mask = fold["test_mask"]

    # 時間排序的訓練+驗證資料
    df_tv = df.loc[train_val_mask].sort_index()
    df_te = df.loc[test_mask].sort_index()

    split_ratio = cfg.get("data", {}).get("train_val_split", 0.8)
    split_idx = int(len(df_tv) * split_ratio)

    tr_idx = df_tv.index[:split_idx]
    va_idx = df_tv.index[split_idx:]
    te_idx = df_te.index

    return tr_idx, va_idx, te_idx

# ========== Dataset for sequence-to-label ==========
# class SeqDataset(Dataset):
#     def __init__(self, X_df, y_s, seq_len: int, scaler: RobustScaler | None = None, device: str = "cuda"):
#         X = X_df.values.astype(np.float32, copy=False)
#         if scaler is not None:
#             X = scaler.transform(X).astype(np.float32, copy=False)
#         y = y_s.values.astype(np.int64, copy=False)

#         self.X = torch.from_numpy(X).contiguous()
#         self.y = torch.from_numpy(y).contiguous()
#         self.L = int(seq_len)
#         # 以序列末端對齊標籤（label 早已是 t→t+1 小時／或你自定義的 horizon）
#         self.idx = np.arange(self.L - 1, len(X) - 1)

#     def __len__(self):
#         return len(self.idx)

#     def __getitem__(self, i):
#         j = self.idx[i]
#         x_seq = self.X[j - self.L + 1: j + 1]  # [T, F]
#         y_val = self.y[j]
#         return x_seq, y_val
    
# # 直接load進VRAM
# class PreloadSeqDataset(torch.utils.data.Dataset):
#     def __init__(self, X_df, y_s, seq_len: int, scaler: RobustScaler | None = None, device="cuda"):
#         X = X_df.values.astype(np.float32, copy=False)
#         if scaler is not None:
#             X = scaler.transform(X).astype(np.float32, copy=False)
#         y = y_s.values.astype(np.int64, copy=False)

#         L = int(seq_len)
#         idx = np.arange(L - 1, len(X) - 1)

#         X_seqs = np.stack([X[j - L + 1: j + 1] for j in idx])  # shape: [N, T, F]
#         y_vals = y[idx]                                        # shape: [N]

#         self.X = torch.tensor(X_seqs, dtype=torch.float32, device=device)
#         self.y = torch.tensor(y_vals, dtype=torch.long, device=device)

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, i):
#         return self.X[i], self.y[i]

# -------------------------
# Preload 到 GPU 的 Dataset
# -------------------------
class PreloadSeqDataset(Dataset):
    def __init__(
        self,
        X_df,
        y_s,
        seq_len: int,
        scaler: Optional[RobustScaler] = None,
        device: str = "cuda",
        label_dtype: Literal["auto", "float", "long"] = "auto",
    ):
        X = X_df.values.astype(np.float32, copy=False)
        if scaler is not None:
            X = scaler.transform(X).astype(np.float32, copy=False)

        if label_dtype == "auto":
            is_float = np.issubdtype(y_s.values.dtype, np.floating)
            y_np = y_s.values.astype(np.float32 if is_float else np.int64, copy=False)
            torch_y_dtype = torch.float32 if is_float else torch.long
        elif label_dtype == "float":
            y_np = y_s.values.astype(np.float32, copy=False)
            torch_y_dtype = torch.float32
        else:  # "long"
            y_np = y_s.values.astype(np.int64, copy=False)
            torch_y_dtype = torch.long

        L = int(seq_len)
        N, M = len(X), len(y_np)
        idx = np.arange(L - 1, min(N, M))

        X_seqs = np.stack([X[j - L + 1 : j + 1] for j in idx])  # shape: [N, T, F]
        y_vals = y_np[idx]                                      # shape: [N]

        self.X = torch.tensor(X_seqs, dtype=torch.float32, device=device)
        self.y = torch.tensor(y_vals, dtype=torch_y_dtype, device=device)
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ========== Fold Generator ==========
class FoldGenerator:
    def __init__(self, dt_index: pd.DatetimeIndex, mode: str = "rolling", start_month: str = "2021-01-01", **kwargs):
        # 移除時區資訊（轉換為 naive），且記錄起始時間戳
        if getattr(dt_index, "tz", None) is not None:
            dt_index = dt_index.tz_convert("Asia/Taipei").tz_localize(None)
        self.dt_index = dt_index
        self.mode = mode
        self.start_ts = pd.Timestamp(start_month)
        self.kwargs = kwargs

        # 2. 全部 folds 的月列表（但只保留起始時間之後的）
        months_all = pd.PeriodIndex(self.dt_index.to_period("M")).unique().sort_values()
        self.months = [m for m in months_all if m.start_time >= self.start_ts]

    def _get_test_months(self, test_freq: str):
        """
        根據 test_freq（例如 'M', '2M', 'Q'）產生要測試的月份。
        """
        if test_freq in {"M", "Q"}:
            return pd.period_range(self.start_ts, self.months[-1], freq=test_freq)
        else:
            # e.g., "2M"、"3M" 等非標準 freq（不是 Pandas 原生 freq）
            # 這裡我們用 manual skip
            months = self.months
            step = int(test_freq.replace("M", ""))  # "2M" → 2
            return months[::step]

    def make_two_month_folds(self):
        """
        回傳 list of dict:
        [{
        'train_val_mask': <bool array>,
        'test_mask': <bool array>,
        'train_val_month': 'YYYY-MM',
        'test_month': 'YYYY-MM'
        }, ...]
        """

        folds = []
        for m in self.months:
            if m.month % 2 == 1:  # 奇數月
                next_month = m + 1
                if next_month in self.months:
                    # ★ 3) 這裡改成用 idx（去時區後），避免 timezone 警告與遮罩錯位
                    train_val_mask = (self.dt_index.to_period("M") == m)
                    test_mask      = (self.dt_index.to_period("M") == next_month)
                    folds.append({
                        'train_val_mask': train_val_mask,
                        'test_mask':      test_mask,
                        'train_val_month': str(m),
                        'test_month':      str(next_month),
                    })
        return folds

    # =========================
    # 2) Anchored folds 產生器（重點）
    # =========================

    def make_anchored_folds(self,
                            embargo_hours: int = 24,         # train→test 的禁區長度（小時）
                            min_train_days: int = 30,        # 每個 fold 至少要有多少天的訓練資料（避免太小）
                            test_freq="M"
                            ):
        """
        產生「Anchored（擴充式）」月度 folds：
        - 對每個測試月份 m：
            * test_mask = (timestamp 的 calendar month == m)
            * train_val_mask = [start_date, test_start - embargo) 的所有資料（擴充式累積）
        - 注意：embargo 以小時為單位，會從 test 月初往回挖空。
        回傳：list[dict]，每個 dict 內含
        - 'train_val_mask': bool array
        - 'test_mask': bool array
        - 'test_month': 'YYYY-MM'
        """

        test_months = self._get_test_months(test_freq)
        folds = []

        for m in test_months:
            # 該月的月初（naive）
            test_start = pd.Timestamp(m.start_time)
            embargo_delta = pd.Timedelta(hours=embargo_hours)
            train_end = test_start - embargo_delta

            # 訓練（擴充式）：從 anchor 起累積到圖上 train_end_exclusive
            train_mask = (self.dt_index >= self.start_ts) & (self.dt_index < train_end)
            test_mask  = (self.dt_index.to_period("M") == m)

            # 至少要有一定天數的訓練資料，避免前期 fold 太短
            if train_mask.sum() == 0:
                continue
            n_days = (pd.DatetimeIndex(self.dt_index[train_mask]).date[-1] -
                    pd.DatetimeIndex(self.dt_index[train_mask]).date[0]).days + 1
            if n_days < min_train_days:
                continue

            # 若該月根本沒資料（或 embargo 刨掉太多），也跳過
            if test_mask.sum() == 0:
                continue

            folds.append({
                "train_val_mask": train_mask,
                "test_mask": test_mask,
                "test_month": str(m)
            })
        return folds

    def make_rolling_folds(self, train_window, embargo_hours, test_freq="M"):
        """
        每一 fold：
        - 訓練集為固定長度（例如過去 3 個月）
        - 測試集為下一個月

        ex. idx = DatetimeIndex(2024-01-01 ~ 2024-05-31)
        => months = pd.PeriodIndex(idx.to_period("M")).unique().sort_values()
        => months = [2024-01, 2024-02, 2024-03, 2024-04, 2024-05]
        """
        test_months = self._get_test_months(test_freq)
        folds = []

        for m in test_months:
            i = self.months.index(m)
            if i < train_window or i + 1 >= len(self.months):
                continue

            train_start = pd.Timestamp(self.months[i - train_window].start_time)
            test_start = pd.Timestamp(m.start_time)
            embargo_delta = pd.Timedelta(hours=embargo_hours)
            train_end = test_start - embargo_delta

            train_mask = (self.dt_index >= train_start) & (self.dt_index < train_end)
            test_mask  = (self.dt_index.to_period("M") == m)

            folds.append({
                'train_val_mask': train_mask,
                'test_mask': test_mask,
                'test_month': str(m)
            })
        return folds


# # ========== Build DataLoaders ==========
# def make_loaders_for_fold(df, feat_cols, target_col, fold, cfg, preload_gpu=True):
#     task_type = _task_type_from_cfg(cfg)
#     is_reg = (task_type == "regression")

#     tr_idx, va_idx, te_idx = split_fold_to_indices(df, fold, cfg)
#     X_tr, y_tr = df.loc[tr_idx, feat_cols], df.loc[tr_idx, target_col]
#     X_va, y_va = df.loc[va_idx, feat_cols], df.loc[va_idx, target_col]
#     X_te, y_te = df.loc[te_idx, feat_cols], df.loc[te_idx, target_col]

#     # --- 特徵縮放器設定 ---
#     scaler_kind = (cfg.get("features", {}) or {}).get("scaler", "robust")
#     if scaler_kind == "robust":
#         scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(10.0, 90.0))
#     elif scaler_kind == "standard":
#         scaler = StandardScaler()
#     elif scaler_kind == "minmax":
#         scaler = MinMaxScaler()
#     else:
#         scaler = None
#     if scaler is not None:
#         scaler.fit(X_tr.values.astype(np.float32, copy=False))

#     # --- 基本設定 ---
#     L = int(cfg["sequence"]["seq_len"])
#     label_dtype = "float" if is_reg else "long"
#     device = cfg.get("train", {}).get("device", "cuda")

#     # --- Dataset 選擇：是否 preload 到 GPU ---
#     dataset_cls = PreloadSeqDataset if preload_gpu else SeqDataset
#     dataset_kwargs = dict(seq_len=L, scaler=scaler, label_dtype=label_dtype)
#     if preload_gpu:
#         dataset_kwargs["device"] = device

#     ds_tr = PreloadSeqDataset(X_tr, y_tr, **dataset_kwargs)
#     ds_va = PreloadSeqDataset(X_va, y_va, **dataset_kwargs)
#     ds_te = PreloadSeqDataset(X_te, y_te, **dataset_kwargs)

#     # --- DataLoader ---
#     bs = int(cfg["train"]["batch_size"])
#     num_workers = 0 if preload_gpu else int(cfg["train"].get("num_workers", 2))  # preload 模式不需 worker
#     pin_mem = not preload_gpu  # preload 模式不用 pin_memory

#     train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=True)
#     val_loader   = DataLoader(ds_va, batch_size=bs, shuffle=False)
#     test_loader  = DataLoader(ds_te, batch_size=bs, shuffle=False)

#     # --- Debug：檢查回歸 label 是否正常 ---
#     if is_reg:
#         xb, yb = next(iter(train_loader))
#         s = float(yb.float().std().item()); u = float(yb.float().mean().item())
#         print(f"[Data] regression labels check: mean={u:.3e}, std={s:.3e}")
#         if s == 0.0:
#             print("[ALERT][dataloader] regression labels std=0 on train batch! 可能被轉整數或切窗錯位。")

#     return train_loader, val_loader, test_loader, {"feat_cols": feat_cols, "target_col": target_col}


# -------------------------
# DataLoader 組裝（永遠 preload）
# -------------------------
def make_loaders_for_fold(df, feat_cols, target_col, fold, cfg, preload_gpu=True):
    # 任務型態
    task_type = _task_type_from_cfg(cfg)
    is_reg = (task_type == "regression")

    # 時序切分（直接依 YAML 的 train_val_split）
    tr_idx, va_idx, te_idx = split_fold_to_indices(df, fold, cfg)
    X_tr, y_tr = df.loc[tr_idx, feat_cols], df.loc[tr_idx, target_col]
    X_va, y_va = df.loc[va_idx, feat_cols], df.loc[va_idx, target_col]
    X_te, y_te = df.loc[te_idx, feat_cols], df.loc[te_idx, target_col]

    # Scaler（只 fit 在 train）
    scaler_kind = (cfg.get("features", {}) or {}).get("scaler", "robust")
    if scaler_kind == "robust":
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(10.0, 90.0))
    elif scaler_kind == "standard":
        scaler = StandardScaler()
    elif scaler_kind == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = None
    if scaler is not None:
        scaler.fit(X_tr.values.astype(np.float32, copy=False))

    # 參數
    L = int(cfg["sequence"]["seq_len"])
    label_dtype = "float" if is_reg else "long"
    device = cfg.get("train", {}).get("device", "cuda")
    bs = int(cfg["train"]["batch_size"])

    # 永遠使用 Preload 到 GPU
    ds_tr = PreloadSeqDataset(X_tr, y_tr, L, scaler=scaler, device=device, label_dtype=label_dtype)
    ds_va = PreloadSeqDataset(X_va, y_va, L, scaler=scaler, device=device, label_dtype=label_dtype)
    ds_te = PreloadSeqDataset(X_te, y_te, L, scaler=scaler, device=device, label_dtype=label_dtype)

    # DataLoader：簡潔版（preload → num_workers=0, pin_memory=False）
    train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=False,  drop_last=False, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(ds_te, batch_size=bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader, {"feat_cols": feat_cols, "target_col": target_col}
