# dataloader.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional


def split_fold_to_indices(df: pd.DataFrame, fold: dict, cfg: dict):
    train_val_mask = fold["train_val_mask"]
    test_mask = fold["test_mask"]

    # 時間排序的訓練+驗證資料
    df_tv = df.loc[train_val_mask].sort_index()
    df_te = df.loc[test_mask].sort_index()

    split_ratio = cfg["cv"]["train_val_split"]
    split_idx = int(len(df_tv) * split_ratio)

    tr_idx = df_tv.index[:split_idx]
    va_idx = df_tv.index[split_idx:]
    te_idx = df_te.index

    return tr_idx, va_idx, te_idx

# -------------------------
#  Dataset
# -------------------------
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


# ========== Fold Generator ==========
class FoldGenerator:
    def __init__(self, dt_index: pd.DatetimeIndex, mode: str = "rolling", start_month: str | None = None, **kwargs):
        """
        僅根據 index 生成 fold。此 index 應已經過 build_features_and_label 切過範圍。
        """
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

    # =========================
    # 1) OddEven folds 產生器
    # =========================
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
    # 2) Anchored folds 產生器
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

    # =========================
    # 3) Rolling folds 產生器
    # =========================

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

# -------------------------
# DataLoader 組裝
# -------------------------
from .scalar import pick_cols_to_scale, _get_scaler, ColumnSubsetScaler

def make_loaders_for_fold(df, feat_cols, target_col, fold, cfg, also_XGB: bool = False):
    """
    依 fold 切出 train/val/test，執行縮放與清理，最後包成三個 DataLoader。
    主要差異點：
      - 若使用 TimeSafeScaler：先 transform_full，再針對每個 split 分別 dropna；
        並在 train split 裁掉 warm-up（scaler.warmup_len()）。
      - 若使用 sklearn 縮放器：先清理 train，再 fit_df(train)；val/test 僅 transform，不看未來。
    """

    task_type = cfg["task"]["type"]
    is_reg = (task_type == "regression")

    # Scaler（只 fit 在 train，否則會洩漏）
    scaler_kind = cfg["sequence"]["scaler"]
    scaler_window = cfg["sequence"]["seq_len"]
    min_frac = cfg["sequence"]["min_frac"]

    # 1) 決定要縮放的欄位（自動跳過 sign-like / 命名 pattern）
    tr_idx, va_idx, te_idx = split_fold_to_indices(df, fold, cfg)
    cols_to_scale = pick_cols_to_scale(df.loc[tr_idx, feat_cols], feat_cols)

    # 2) 建立縮放器
    scaler = _get_scaler(scaler_kind, window=scaler_window, min_frac=min_frac)

    # 3) 時間安全縮放：先對整段 df 做 transform_full（只動 cols_to_scale）
    if hasattr(scaler, "is_timesafe") and scaler.is_timesafe:
        df_scaled = scaler.transform_full(df, cols_to_scale=cols_to_scale)
        work_df = df_scaled
        sklearn_scaler = None
    else:
        work_df = df
        sklearn_scaler = None if scaler is None else ColumnSubsetScaler(
            scaler, all_cols=feat_cols, cols_to_scale=cols_to_scale
        )


    # 4) 切 train/val/test（在時間安全模式已經先轉好；sklearn 模式稍後 fit+transform）
    
    X_tr, y_tr = work_df.loc[tr_idx, feat_cols], work_df.loc[tr_idx, target_col]
    X_va, y_va = work_df.loc[va_idx, feat_cols], work_df.loc[va_idx, target_col]
    X_te, y_te = work_df.loc[te_idx, feat_cols], work_df.loc[te_idx, target_col]

    def _clean_split(X_df, y_s):
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        valid = X_df.notna().all(axis=1) & y_s.notna()
        return X_df.loc[valid], y_s.loc[valid]

    # 清理nan
    X_tr, y_tr = _clean_split(X_tr, y_tr)
    X_va, y_va = _clean_split(X_va, y_va)
    X_te, y_te = _clean_split(X_te, y_te)

    # 5) sklearn 縮放：只用 train 擬合，SeqDataset 會在 GPU 前再 transform（不洩漏、且只動 cols_to_scale）
    if sklearn_scaler is not None:
        sklearn_scaler.fit_df(X_tr)

     # 6) 建 Dataset / Loader（和你原本一致）
    L = int(cfg["sequence"]["seq_len"])
    label_dtype = "float" if is_reg else "long"
    runtime_device = cfg["device"]
    bs = int(cfg["train"]["batch_size"])

    # 是否預先把整個 Dataset 放上 GPU（容易 OOM）；預設 False → 留在 CPU，再在 trainer 逐 batch 搬到 GPU
    preload_to_gpu = bool(cfg.get("sequence", {}).get("preload_to_gpu", False))
    ds_device = runtime_device if (preload_to_gpu and runtime_device == "cuda") else "cpu"

    stride = cfg["sequence"]["stride"]
    anchor = int(cfg["sequence"]["stride_anchor"]) % stride

    ds_tr = SeqDataset(X_tr, y_tr, L, scaler=sklearn_scaler, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)
    ds_va = SeqDataset(X_va, y_va, L, scaler=sklearn_scaler, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)
    ds_te = SeqDataset(X_te, y_te, L, scaler=sklearn_scaler, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)

    # DataLoader：若 Dataset 在 CPU 且 runtime 在 CUDA，開啟 pin_memory + num_workers 加速搬運
    pin = (ds_device == "cpu" and runtime_device == "cuda")
    num_workers = 10 if pin else 0
    train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(ds_te, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)

    info = {"feat_cols": feat_cols, "target_col": target_col}


    # 7) 也把縮放資訊給 XGB 分支
    if also_XGB:
        Xtr = X_tr.values.astype(np.float32, copy=False)
        Xva = X_va.values.astype(np.float32, copy=False)
        Xte = X_te.values.astype(np.float32, copy=False)
        if sklearn_scaler is not None:
            Xtr = sklearn_scaler.transform(Xtr)
            Xva = sklearn_scaler.transform(Xva)
            Xte = sklearn_scaler.transform(Xte)
        y_dtype = np.float32 if is_reg else np.int64
        info["XGB"] = {
            "X_tr": Xtr, "y_tr": y_tr.values.astype(y_dtype, copy=False),
            "X_va": Xva, "y_va": y_va.values.astype(y_dtype, copy=False),
            "X_te": Xte, "y_te": y_te.values.astype(y_dtype, copy=False),
            "scaler": sklearn_scaler,
            "cols_to_scale": cols_to_scale,
        }

    return train_loader, val_loader, test_loader, info



# def make_loaders_for_fold(df, feat_cols, target_col, fold, cfg, also_XGB: bool = False):
#     """
#     依 fold 切出 train/val/test，執行縮放與清理，最後包成三個 DataLoader。

#     event_tbm 模式：
#       - 保留完整 15m 連續網格（X 不砍到只剩事件列）
#       - 以 y.notna()（或 y>=0）的位置當「事件錨點」取樣（SeqDataset.anchor_positions）
#       - 每個樣本仍用長度 L 的連續滑窗，避免時間不規則

#     time-driven 模式：
#       - 舊邏輯：依 stride/anchor 規律抽樣
#       - 會過濾掉 y 為 NaN 的列

#     依賴：
#       - split_fold_to_indices(df, fold, cfg)
#       - pick_cols_to_scale(train_df, feat_cols)
#       - _get_scaler(kind, window, min_frac)
#       - ColumnSubsetScaler(sklearn_scaler, all_cols, cols_to_scale)
#       - SeqDataset(..., anchor_positions=..., min_valid_ratio=...)
#     """
#     task_type = str(cfg["task"]["type"]).lower()
#     is_reg = (task_type == "regression")
#     is_event = str(cfg["label"]["mode"]).lower() == "event_tbm"

#     # === Scaler/窗口設定 ===
#     scaler_kind   = cfg["sequence"]["scaler"]
#     seq_len       = int(cfg["sequence"]["seq_len"])
#     min_valid     = float(cfg["sequence"]["min_frac"])  # 用於 SeqDataset.min_valid_ratio
#     stride_cfg    = int(cfg["sequence"]["stride"])
#     anchor_cfg    = int(cfg["sequence"]["stride_anchor"]) % max(1, stride_cfg)

#     # === 1) 切 fold & 選要縮放的欄位 ===
#     tr_idx, va_idx, te_idx = split_fold_to_indices(df, fold, cfg)
#     cols_to_scale = pick_cols_to_scale(df.loc[tr_idx, feat_cols], feat_cols)

#     # === 2) 建立縮放器 ===
#     scaler = _get_scaler(scaler_kind, window=seq_len, min_frac=min_valid)

#     # === 3) 時間安全縮放：先對整段 df 做 transform_full（只動 cols_to_scale） ===
#     if hasattr(scaler, "is_timesafe") and scaler.is_timesafe:
#         df_scaled = scaler.transform_full(df, cols_to_scale=cols_to_scale)
#         work_df = df_scaled
#         sklearn_scaler = None
#     else:
#         work_df = df
#         sklearn_scaler = None if scaler is None else ColumnSubsetScaler(
#             scaler, all_cols=feat_cols, cols_to_scale=cols_to_scale
#         )

#     # === 4) 切 train/val/test（event 模式先保留 y 的 NaN） ===
#     X_tr, y_tr = work_df.loc[tr_idx, feat_cols], work_df.loc[tr_idx, target_col]
#     X_va, y_va = work_df.loc[va_idx, feat_cols], work_df.loc[va_idx, target_col]
#     X_te, y_te = work_df.loc[te_idx, feat_cols], work_df.loc[te_idx, target_col]

#     # === 5) 清理 NaN/Inf ===
#     def _clean_split(X_df, y_s, *, drop_y_na: bool):
#         # 先把 ±inf 轉 NaN
#         X_df = X_df.replace([np.inf, -np.inf], np.nan)
#         # event 模式：保留 y 的 NaN（非事件），但對特徵做單向 ffill（只用過去補）
#         if not drop_y_na:
#             X_df = X_df.ffill()
#             return X_df, y_s
#         # time-driven：特徵完整且 y 有值才保留
#         valid = X_df.notna().all(axis=1) & y_s.notna()
#         return X_df.loc[valid], y_s.loc[valid]

#     X_tr, y_tr = _clean_split(X_tr, y_tr, drop_y_na=not is_event)
#     X_va, y_va = _clean_split(X_va, y_va, drop_y_na=not is_event)
#     X_te, y_te = _clean_split(X_te, y_te, drop_y_na=not is_event)

#     # === 6) sklearn 縮放：僅用 train 的「特徵完整列」fit（避免洩漏） ===
#     if sklearn_scaler is not None:
#         fit_idx = X_tr.notna().all(axis=1)
#         if fit_idx.any():
#             sklearn_scaler.fit_df(X_tr.loc[fit_idx])
#         else:
#             print("[make_loaders_for_fold] WARNING: no valid rows to fit scaler on train split.")

#     # === 7) Dataset / DataLoader ===
#     runtime_device = cfg["device"]
#     bs = int(cfg["train"]["batch_size"])
#     preload_to_gpu = bool(cfg.get("sequence", {}).get("preload_to_gpu", False))
#     ds_device = runtime_device if (preload_to_gpu and runtime_device == "cuda") else "cpu"

#     # label dtype
#     if is_reg:
#         label_dtype = "float"
#     else:
#         if is_event:
#             # y 的 NaN → -1（非事件），真正取樣由 anchor_positions 控制
#             y_tr = y_tr.fillna(-1).astype(np.int64)
#             y_va = y_va.fillna(-1).astype(np.int64)
#             y_te = y_te.fillna(-1).astype(np.int64)
#         label_dtype = "long"

#     # 事件錨點 / stride
#     if is_event:
#         anc_tr = np.where(y_tr.values >= 0)[0]  # 事件列：0/1
#         anc_va = np.where(y_va.values >= 0)[0]
#         anc_te = np.where(y_te.values >= 0)[0]
#         use_stride, use_anchor = 1, 0  # 事件驅動建議 stride=1
#     else:
#         anc_tr = anc_va = anc_te = None
#         use_stride, use_anchor = stride_cfg, anchor_cfg

#     # 建 Dataset（需使用改過、支援 anchor_positions 與 min_valid_ratio 的 SeqDataset）
#     ds_tr = SeqDataset(
#         X_tr, y_tr, seq_len,
#         scaler=sklearn_scaler,
#         device=ds_device,
#         label_dtype=label_dtype,
#         stride=use_stride,
#         anchor=use_anchor,
#         anchor_positions=anc_tr,
#         min_valid_ratio=min_valid,
#     )
#     ds_va = SeqDataset(
#         X_va, y_va, seq_len,
#         scaler=sklearn_scaler,
#         device=ds_device,
#         label_dtype=label_dtype,
#         stride=use_stride,
#         anchor=use_anchor,
#         anchor_positions=anc_va,
#         min_valid_ratio=min_valid,
#     )
#     ds_te = SeqDataset(
#         X_te, y_te, seq_len,
#         scaler=sklearn_scaler,
#         device=ds_device,
#         label_dtype=label_dtype,
#         stride=use_stride,
#         anchor=use_anchor,
#         anchor_positions=anc_te,
#         min_valid_ratio=min_valid,
#     )

#     # DataLoader（CPU→CUDA 時開 pin_memory / workers）
#     pin = (ds_device == "cpu" and runtime_device == "cuda")
#     num_workers = 10 if pin else 0
#     train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=False, drop_last=False,
#                               num_workers=num_workers, pin_memory=pin)
#     val_loader   = DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False,
#                               num_workers=num_workers, pin_memory=pin)
#     test_loader  = DataLoader(ds_te, batch_size=bs, shuffle=False, drop_last=False,
#                               num_workers=num_workers, pin_memory=pin)

#     info = {"feat_cols": feat_cols, "target_col": target_col}

#     # === 8) XGB（tabular）資料：event 模式只取事件列；非 event 取全部 ===
#     if also_XGB:
#         if is_event:
#             mtr = (y_tr.values >= 0)
#             mva = (y_va.values >= 0)
#             mte = (y_te.values >= 0)
#             Xtr = X_tr.loc[mtr].values.astype(np.float32, copy=False)
#             Xva = X_va.loc[mva].values.astype(np.float32, copy=False)
#             Xte = X_te.loc[mte].values.astype(np.float32, copy=False)
#             ytr = y_tr.loc[mtr].values.astype(np.int64, copy=False)
#             yva = y_va.loc[mva].values.astype(np.int64, copy=False)
#             yte = y_te.loc[mte].values.astype(np.int64, copy=False)
#         else:
#             Xtr = X_tr.values.astype(np.float32, copy=False)
#             Xva = X_va.values.astype(np.float32, copy=False)
#             Xte = X_te.values.astype(np.float32, copy=False)
#             y_dtype = np.float32 if is_reg else np.int64
#             ytr = y_tr.values.astype(y_dtype, copy=False)
#             yva = y_va.values.astype(y_dtype, copy=False)
#             yte = y_te.values.astype(y_dtype, copy=False)

#         if sklearn_scaler is not None:
#             Xtr = sklearn_scaler.transform(Xtr)
#             Xva = sklearn_scaler.transform(Xva)
#             Xte = sklearn_scaler.transform(Xte)

#         info["XGB"] = {
#             "X_tr": Xtr, "y_tr": ytr,
#             "X_va": Xva, "y_va": yva,
#             "X_te": Xte, "y_te": yte,
#             "scaler": sklearn_scaler,
#             "cols_to_scale": cols_to_scale,
#         }

#     # === 9) 友善警示 ===
#     if len(train_loader) == 0:
#         n_ev = len(anc_tr) if is_event else -1
#         print(f"[make_loaders_for_fold] WARNING: empty train loader | "
#               f"is_event={is_event}, events_in_train={n_ev}, "
#               f"seq_len={seq_len}, min_valid_ratio={min_valid}. "
#               f"Consider lowering seq_len/min_valid or enlarging CV window.")

#     return train_loader, val_loader, test_loader, info


# ===============================
# BACKUP (commented): Original time-driven implementations
# ===============================
# NOTE: Kept here per user request as an inline backup. Do not edit.

# --- BACKUP: Original time-driven SeqDataset ---
class SeqDataset(Dataset):
    def __init__(
        self,
        X_df,
        y_s,
        seq_len: int,
        scaler=None,
        device: str = "cuda",
        label_dtype: Literal["auto", "float", "long"] = "auto",
        stride: int = 1,
        anchor: int = 0,
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
        L = int(seq_len)
        N, M = len(X), len(y_np)
        stride = max(1, int(stride))
        anchor = int(anchor) % stride
        start = (L - 1) + anchor
        stop = min(N, M)
        if start >= stop:
            start = L - 1
        idx = np.arange(start=start, stop=stop, step=stride, dtype=int)
        X_seqs = np.stack([X[j - L + 1: j + 1] for j in idx]) if len(idx) else np.empty((0, L, X.shape[1]), np.float32)
        y_vals = y_np[idx] if len(idx) else np.empty((0,), y_np.dtype)
        self.X = torch.tensor(X_seqs, dtype=torch.float32, device=device)
        self.y = torch.tensor(y_vals, dtype=torch_y_dtype, device=device)
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# --- BACKUP: Original make_loaders_for_fold (time-driven) ---
# from .scalar import pick_cols_to_scale, _get_scaler, ColumnSubsetScaler
# def make_loaders_for_fold(df, feat_cols, target_col, fold, cfg, also_XGB: bool = False):
#     task_type = cfg["task"]["type"]
#     is_reg = (task_type == "regression")
#     scaler_kind = cfg["sequence"]["scaler"]
#     scaler_window = cfg["sequence"]["seq_len"]
#     min_frac = cfg["sequence"]["min_frac"]
#     tr_idx, va_idx, te_idx = split_fold_to_indices(df, fold, cfg)
#     cols_to_scale = pick_cols_to_scale(df.loc[tr_idx, feat_cols], feat_cols)
#     scaler = _get_scaler(scaler_kind, window=scaler_window, min_frac=min_frac)
#     if hasattr(scaler, "is_timesafe") and scaler.is_timesafe:
#         df_scaled = scaler.transform_full(df, cols_to_scale=cols_to_scale)
#         work_df = df_scaled
#         sklearn_scaler = None
#     else:
#         work_df = df
#         sklearn_scaler = None if scaler is None else ColumnSubsetScaler(
#             scaler, all_cols=feat_cols, cols_to_scale=cols_to_scale
#         )
#     X_tr, y_tr = work_df.loc[tr_idx, feat_cols], work_df.loc[tr_idx, target_col]
#     X_va, y_va = work_df.loc[va_idx, feat_cols], work_df.loc[va_idx, target_col]
#     X_te, y_te = work_df.loc[te_idx, feat_cols], work_df.loc[te_idx, target_col]
#     def _clean_split(X_df, y_s):
#         X_df = X_df.replace([np.inf, -np.inf], np.nan)
#         valid = X_df.notna().all(axis=1) & y_s.notna()
#         return X_df.loc[valid], y_s.loc[valid]
#     X_tr, y_tr = _clean_split(X_tr, y_tr)
#     X_va, y_va = _clean_split(X_va, y_va)
#     X_te, y_te = _clean_split(X_te, y_te)
#     if sklearn_scaler is not None:
#         sklearn_scaler.fit_df(X_tr)
#     L = int(cfg["sequence"]["seq_len"])
#     label_dtype = "float" if is_reg else "long"
#     runtime_device = cfg["device"]
#     bs = int(cfg["train"]["batch_size"])
#     preload_to_gpu = bool(cfg.get("sequence", {}).get("preload_to_gpu", False))
#     ds_device = runtime_device if (preload_to_gpu and runtime_device == "cuda") else "cpu"
#     stride = cfg["sequence"]["stride"]
#     anchor = int(cfg["sequence"]["stride_anchor"]) % stride
#     ds_tr = SeqDataset(X_tr, y_tr, L, scaler=sklearn_scaler, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)
#     ds_va = SeqDataset(X_va, y_va, L, scaler=sklearn_scaler, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)
#     ds_te = SeqDataset(X_te, y_te, L, scaler=sklearn_scaler, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)
#     pin = (ds_device == "cpu" and runtime_device == "cuda")
#     num_workers = 10 if pin else 0
#     train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
#     val_loader   = DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
#     test_loader  = DataLoader(ds_te, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
#     info = {"feat_cols": feat_cols, "target_col": target_col}
#     if also_XGB:
#         Xtr = X_tr.values.astype(np.float32, copy=False)
#         Xva = X_va.values.astype(np.float32, copy=False)
#         Xte = X_te.values.astype(np.float32, copy=False)
#         if sklearn_scaler is not None:
#             Xtr = sklearn_scaler.transform(Xtr)
#             Xva = sklearn_scaler.transform(Xva)
#             Xte = sklearn_scaler.transform(Xte)
#         y_dtype = np.float32 if is_reg else np.int64
#         info["XGB"] = {
#             "X_tr": Xtr, "y_tr": y_tr.values.astype(y_dtype, copy=False),
#             "X_va": Xva, "y_va": y_va.values.astype(y_dtype, copy=False),
#             "X_te": Xte, "y_te": y_te.values.astype(y_dtype, copy=False),
#             "scaler": sklearn_scaler,
#             "cols_to_scale": cols_to_scale,
#         }
#     return train_loader, val_loader, test_loader, info
