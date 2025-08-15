
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

# ========== Dataset for sequence-to-label ==========
class SeqDataset(Dataset):
    def __init__(self, X_df, y_s, seq_len: int, scaler: RobustScaler | None = None, device: str = "cuda"):
        X = X_df.values.astype(np.float32, copy=False)
        if scaler is not None:
            X = scaler.transform(X).astype(np.float32, copy=False)
        y = y_s.values.astype(np.int64, copy=False)

        self.X = torch.from_numpy(X).contiguous()
        self.y = torch.from_numpy(y).contiguous()
        self.L = int(seq_len)
        # 以序列末端對齊標籤（label 早已是 t→t+1 小時／或你自定義的 horizon）
        self.idx = np.arange(self.L - 1, len(X) - 1)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x_seq = self.X[j - self.L + 1: j + 1]  # [T, F]
        y_val = self.y[j]
        return x_seq, y_val
    
# 直接load進VRAM
class PreloadSeqDataset(torch.utils.data.Dataset):
    def __init__(self, X_df, y_s, seq_len: int, scaler: RobustScaler | None = None, device="cuda"):
        X = X_df.values.astype(np.float32, copy=False)
        if scaler is not None:
            X = scaler.transform(X).astype(np.float32, copy=False)
        y = y_s.values.astype(np.int64, copy=False)

        L = int(seq_len)
        idx = np.arange(L - 1, len(X) - 1)

        X_seqs = np.stack([X[j - L + 1: j + 1] for j in idx])  # shape: [N, T, F]
        y_vals = y[idx]                                        # shape: [N]

        self.X = torch.tensor(X_seqs, dtype=torch.float32, device=device)
        self.y = torch.tensor(y_vals, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.y)

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


# ========== Build DataLoaders ==========
def make_loaders_for_fold(df, feat_cols, label_col, fold, cfg, preload_gpu=False):
    """
    df: 含特徵與標籤的 DataFrame（索引為時間）
    feat_cols: 使用哪些欄位做 X
    label_col: 目標欄位名（分類請用 int 標籤；回歸則自行調整 Dataset）
    fold: 來自 make_anchored_monthly_folds(...) 的 dict
    cfg: 統一設定（含 seq_len/batch_size/train_val_split/num_workers 等）
    """
    if preload_gpu:
        DatasetClass = PreloadSeqDataset
        dl_kwargs = dict(batch_size=int(cfg["train"]["batch_size"]))
    else:
        DatasetClass = SeqDataset
        # 建議在 Linux/多核下開 workers、prefetch，加速供應
        dl_kwargs = dict(batch_size=int(cfg["train"]["batch_size"]), 
                         num_workers=int(cfg["cv"]["num_workers"]), 
                         pin_memory=bool(cfg["cv"]["pin_memory"]), 
                         prefetch_factor=int(cfg["cv"]["prefetch_factor"]), 
                         persistent_workers=True)
    

    device = torch.device("cuda") if preload_gpu else "cpu"

    seq_cfg = cfg["sequence"]["seq_len"]
    used_seq_len = int(np.median(seq_cfg)) if isinstance(seq_cfg, list) else int(seq_cfg)

    tv_mask, ts_mask = fold["train_val_mask"], fold["test_mask"]
    df_tv, df_ts = df.loc[tv_mask], df.loc[ts_mask]
    n = len(df_tv)
    split = int(n * cfg["cv"]["train_val_split"])
    df_tr, df_va = df_tv.iloc[:split], df_tv.iloc[split:]

    scaler = RobustScaler() if cfg["sequence"]["scaler"] == "RobustScaler" else None
    if scaler is not None:
        scaler.fit(df_tr[feat_cols].values)

    ds_tr = DatasetClass(df_tr[feat_cols], df_tr[label_col], seq_len=used_seq_len, scaler=scaler, device=device)
    ds_va = DatasetClass(df_va[feat_cols], df_va[label_col], seq_len=used_seq_len, scaler=scaler, device=device)
    ds_te = DatasetClass(df_ts[feat_cols], df_ts[label_col], seq_len=used_seq_len, scaler=scaler, device=device)

    # 空資料保護
    assert len(ds_tr) > 0 and len(ds_va) > 0 and len(ds_te) > 0, \
        f"Empty dataset with seq_len={used_seq_len} (fold={fold.get('test_month')})"
    
    tr_loader = DataLoader(ds_tr, shuffle=True,  **dl_kwargs)
    va_loader = DataLoader(ds_va, shuffle=False, **dl_kwargs)
    te_loader = DataLoader(ds_te, shuffle=False, **dl_kwargs)
    return tr_loader, va_loader, te_loader, used_seq_len