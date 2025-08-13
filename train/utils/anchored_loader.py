import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from dataclasses import dataclass

# =========================
# 1) 時序 Dataset（你原本的 CSV 版）
# =========================
class SeqDataset(Dataset):
    def __init__(self, X_df, y_s, seq_len: int, scaler: RobustScaler | None = None):
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


# =========================
# 2) Anchored folds 產生器（重點）
# =========================

def make_anchored_monthly_folds(dt_index: pd.DatetimeIndex,
                                start_date: str = "2021-01-01",  # 錨點（anchor）起始
                                # test_freq: str = "M",            # 測試頻率：'M'=每月一fold（依 calendar month）
                                embargo_hours: int = 24,         # train→test 的禁區長度（小時）
                                min_train_days: int = 30        # 每個 fold 至少要有多少天的訓練資料（避免太小）
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
    idx = pd.DatetimeIndex(dt_index).tz_localize(None) if dt_index.tz is not None else pd.DatetimeIndex(dt_index)
    dt_index = pd.DatetimeIndex(dt_index).tz_localize(None)
    start_ts = pd.Timestamp(start_date)
    keep = idx >= start_ts
    idx = idx[keep]

    # 可用月份清單
    months = pd.PeriodIndex(idx.to_period("M")).unique().sort_values()
    folds = []

    for m in months:
        # 測試區間：整個月
        test_mask = (pd.PeriodIndex(dt_index.to_period("M")) == m)

        # 該月的月初（naive）
        test_start = pd.Timestamp(m.start_time)
        embargo_delta = pd.Timedelta(hours=embargo_hours)
        train_end_exclusive = test_start - embargo_delta  # 直到這個時間點「之前」為止

        # 訓練（擴充式）：從 anchor 起累積到圖上 train_end_exclusive
        train_val_mask = (pd.DatetimeIndex(dt_index) >= start_ts) & (pd.DatetimeIndex(dt_index) < train_end_exclusive)

        # 至少要有一定天數的訓練資料，避免前期 fold 太短
        if train_val_mask.sum() == 0:
            continue
        n_days = (pd.DatetimeIndex(dt_index[train_val_mask]).date[-1] -
                  pd.DatetimeIndex(dt_index[train_val_mask]).date[0]).days + 1
        if n_days < min_train_days:
            continue

        # 若該月根本沒資料（或 embargo 刨掉太多），也跳過
        if test_mask.sum() == 0:
            continue

        folds.append({
            "train_val_mask": train_val_mask,
            "test_mask": test_mask,
            "test_month": str(m)
        })
    return folds


# =========================
# 3) 你的 CSV 版 Loader 函式（原樣保留 + 小修註解）
# =========================
def make_loaders_for_fold(df, feat_cols, label_col, fold, cfg):
    """
    df: 含特徵與標籤的 DataFrame（索引為時間）
    feat_cols: 使用哪些欄位做 X
    label_col: 目標欄位名（分類請用 int 標籤；回歸則自行調整 Dataset）
    fold: 來自 make_anchored_monthly_folds(...) 的 dict
    cfg: 統一設定（含 seq_len/batch_size/train_val_split/num_workers 等）
    """
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

    ds_tr = SeqDataset(df_tr[feat_cols], df_tr[label_col], seq_len=used_seq_len, scaler=scaler)
    ds_va = SeqDataset(df_va[feat_cols], df_va[label_col], seq_len=used_seq_len, scaler=scaler)
    ds_te = SeqDataset(df_ts[feat_cols], df_ts[label_col], seq_len=used_seq_len, scaler=scaler)

    # 空資料保護
    assert len(ds_tr) > 0 and len(ds_va) > 0 and len(ds_te) > 0, \
        f"Empty dataset with seq_len={used_seq_len} (fold={fold.get('test_month')})"

    bs = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["cv"]["num_workers"])
    pin_memory  = bool(cfg["cv"]["pin_memory"])

    # 建議在 Linux/多核下開 workers、prefetch，加速供應
    dl_kwargs = dict(batch_size=bs, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    if num_workers > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=8))

    tr_loader = DataLoader(ds_tr, shuffle=True,  **dl_kwargs)
    va_loader = DataLoader(ds_va, shuffle=False, **dl_kwargs)
    te_loader = DataLoader(ds_te, shuffle=False, **dl_kwargs)
    return tr_loader, va_loader, te_loader, used_seq_len



def make_rolling_monthly_folds(dt_index, train_window, embargo_hours):
    """
    每一 fold：
    - 訓練集為固定長度（例如過去 3 個月）
    - 測試集為下一個月
    """
    idx = pd.DatetimeIndex(dt_index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Taipei").tz_localize(None)

    months = pd.PeriodIndex(idx.to_period("M")).unique().sort_values()

    # ex. idx = DatetimeIndex(2024-01-01 ~ 2024-05-31)
    # => months = pd.PeriodIndex(idx.to_period("M")).unique().sort_values()
    # => months = [2024-01, 2024-02, 2024-03, 2024-04, 2024-05]

    folds = []

    train_window = train_window  # e.g., 3
    embargo_hours = embargo_hours

    for i in range(train_window, len(months) - 1):
        train_start = pd.Timestamp(months[i - train_window].start_time) # 把 month轉成timestamp
        test_month = months[i]

        test_start = pd.Timestamp(test_month.start_time)
        test_end = pd.Timestamp(test_month.end_time)

        embargo_delta = pd.Timedelta(hours=embargo_hours)
        train_end = test_start - embargo_delta

        train_mask = (idx >= train_start) & (idx < train_end)
        test_mask = (pd.PeriodIndex(dt_index.to_period("M")) == test_month)

        folds.append({
            "train_val_mask": train_mask,
            "test_mask": test_mask,
            "test_month": str(test_month)
        })

    return folds

