import os
import pandas as pd, numpy as np
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

def _ensure_datetime_index(obj):
    """
    支援多種 ts 形態：
    - list[str]（ISO 字串，含或不含時區）
    - list[int] / np.ndarray[int]（ms 或 ns since epoch）
    - torch.Tensor[int]（同上）
    會盡量還原成 tz-aware（若原本含 +08:00 就保留），否則 naive。
    """
    # torch.Tensor -> numpy
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()

    # 字串列表
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], str):
        return pd.DatetimeIndex(pd.to_datetime(obj, utc=False, errors="raise"))

    # 整數（ms 或 ns）
    if isinstance(obj, (list, tuple, np.ndarray)) and len(obj) > 0 and np.issubdtype(np.array(obj).dtype, np.integer):
        arr = np.asarray(obj)
        # 判斷是 ms 還是 ns：用量級估
        # 2021年 epoch 大約 1.6e12(ms) / 1.6e18(ns)
        if arr.mean() < 1e14:
            # ms
            return pd.DatetimeIndex(pd.to_datetime(arr, unit="ms", utc=True)).tz_convert("Asia/Taipei")
        else:
            # ns
            return pd.DatetimeIndex(pd.to_datetime(arr, unit="ns", utc=True)).tz_convert("Asia/Taipei")

    # 直接丟給 Pandas 試試
    return pd.DatetimeIndex(obj)

def load_pt_cache(pt_path: str):
    """
    讀取 .pt 快取，預期內容：
    {"X": FloatTensor [N,F], "y": LongTensor [N], "ts": list[str]/int/ms/ns, "feat_cols": list[str]}
    """
    blob = torch.load(pt_path, map_location="cpu")
    X_cpu = blob["X"].contiguous()  # [N, F] float32
    y_cpu = blob["y"].contiguous()  # [N]    int64
    ts = _ensure_datetime_index(blob["ts"]) if "ts" in blob else None
    feat_cols = list(blob["feat_cols"]) if "feat_cols" in blob else None
    return X_cpu, y_cpu, ts, feat_cols

# ====== 兩個月為一 fold：奇數月 train/val → 偶數月 test ======
def make_two_month_folds(dt_index, start_month_str: str = '2021-01-01'):
    """
    回傳 list of dict:
    [{
      'train_val_mask': <bool array>,
      'test_mask': <bool array>,
      'train_val_month': 'YYYY-MM',
      'test_month': 'YYYY-MM'
    }, ...]
    """
    # 1) 統一去時區 → naive
    idx = pd.DatetimeIndex(dt_index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Taipei").tz_localize(None)

    # 2) 只取起始月之後
    start_ts = pd.Timestamp(start_month_str)
    mask_after = idx >= start_ts
    months = pd.PeriodIndex(idx[mask_after].to_period("M")).unique().sort_values()

    folds = []
    for m in months:
        if m.month % 2 == 1:  # 奇數月
            next_month = m + 1
            if next_month in months:
                # ★ 3) 這裡改成用 idx（去時區後），避免 timezone 警告與遮罩錯位
                train_val_mask = (idx.to_period("M") == m)
                test_mask      = (idx.to_period("M") == next_month)
                folds.append({
                    'train_val_mask': train_val_mask,
                    'test_mask':      test_mask,
                    'train_val_month': str(m),
                    'test_month':      str(next_month),
                })
    return folds


# Anchored folds
def make_anchored_folds(
    dt_index: pd.DatetimeIndex,
    start_date: str = "2021-01-01",
    test_months: List[str] | None = None,   # 例如 ["2021-02","2021-04", ...]；None 則用所有偶數月
    embargo_hours: int = 96,                # 覆蓋 max(seq_len, max_indicator_window) + buffer
    val_days: int = 14,                     # 驗證窗長度（天）
    tz: str = "Asia/Taipei",
) -> List[Dict]:
    """
    回傳 folds: list[dict]，每個 dict 內含：
      - 'test_month': 'YYYY-MM'
      - 'train_mask', 'val_mask', 'embargo_mask', 'test_mask': np.bool_ array（與 dt_index 對齊）
      - 'ranges': {train:(s,e), val:(s,e), embargo:(s,e), test:(s,e)}（皆為 tz-aware Timestamp）
    """
    idx = pd.DatetimeIndex(dt_index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(tz)
    else:
        idx = idx.tz_localize(tz)
    idx_naive = idx.tz_localize(None)

    # 1. 可用的月份
    start_ts = pd.Timestamp(start_date, tz=tz)
    use_mask = idx >= start_ts
    months = pd.PeriodIndex(idx_naive[use_mask].to_period("M")).unique().sort_values()

    if test_months is None:
        # 預設：所有偶數月當作測試月
        test_months = [str(m) for m in months if m.month % 2 == 0]

    folds = []
    for m_str in test_months:
        m = pd.Period(m_str, freq="M")
        # 測試月起訖（tz-aware）
        test_start = pd.Timestamp(f"{m.start_time}", tz=tz)
        test_end   = pd.Timestamp(f"{(m+1).start_time}", tz=tz)

        # 驗證 & Embargo 邊界
        embargo_end   = test_start                                # = Test.start
        embargo_start = embargo_end - pd.Timedelta(hours=embargo_hours)
        val_end       = embargo_start
        val_start     = val_end - pd.Timedelta(days=val_days)

        # 早期資料不足時，拉回到 start_ts
        train_start = start_ts
        train_end   = val_start

        if not (train_start < train_end < val_end < embargo_end <= test_end):
            # 若早期月份資料太少導致無法切出完整 train/val/embargo/test，直接跳過
            continue

        # --- 2) 轉成 naive 再遮罩（與 idx_naive 同一座標系） ---
        def n(ts): return ts.tz_convert(tz).tz_localize(None)

        tr_mask = (idx_naive >= n(train_start))   & (idx_naive < n(train_end))
        va_mask = (idx_naive >= n(val_start))     & (idx_naive < n(val_end))
        em_mask = (idx_naive >= n(embargo_start)) & (idx_naive < n(embargo_end))
        te_mask = (idx_naive >= n(test_start))    & (idx_naive < n(test_end))

        # 空集保護
        if tr_mask.sum()==0 or va_mask.sum()==0 or te_mask.sum()==0:
            continue

        folds.append({
            "test_month": m_str,
            "train_mask": tr_mask,
            "val_mask": va_mask,
            "embargo_mask": em_mask,
            "test_mask": te_mask,
            "ranges": {
                "train":   (train_start,   train_end),
                "val":     (val_start,     val_end),
                "embargo": (embargo_start, embargo_end),
                "test":    (test_start,    test_end),
            },
        })
    return folds


# ====== Dataset / Loader ======
class SeqDataset(Dataset):
    def __init__(self, X_df, y_s, seq_len: int, scaler: RobustScaler | None = None):
        # numpy → (可選)標準化 → 一次性轉成 torch tensor（避免 __getitem__ 每次轉）
        X = X_df.values.astype(np.float32, copy=False)
        if scaler is not None:
            X = scaler.transform(X).astype(np.float32, copy=False)
        y = y_s.values.astype(np.int64, copy=False)

        # ★ 一次性轉成 tensor，後面 getitem 直接切片
        self.X = torch.from_numpy(X).contiguous()          # [N, F], float32 (CPU)
        self.y = torch.from_numpy(y).contiguous()          # [N],    int64   (CPU)
        self.L = int(seq_len)
        # 以序列末端對齊標籤（label 早已是 t→t+1 小時）
        self.idx = np.arange(self.L - 1, len(X) - 1)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        # 直接在 CPU tensor 上切片，DataLoader(pin_memory=True) 會幫你做 pinned memory
        x_seq = self.X[j - self.L + 1: j + 1]        # [T, F] float32
        y_val = self.y[j]                             # scalar int64
        return x_seq, y_val


def make_loaders_for_fold(df, feat_cols, label_col, fold, cfg):
    # seq_len（若 list → 交給 Optuna；若單值 → 固定）
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

    # 空資料保護（邏輯檢查）
    assert len(ds_tr) > 0 and len(ds_va) > 0 and len(ds_te) > 0, \
        f"Empty dataset with seq_len={used_seq_len}"
    
    # ★ DataLoader 參數：提高資料供應能力
    bs = cfg["train"]["batch_size"]

    # 合理預設：多核 CPU 就用 8~16；少核機器就取可用核-2
    num_workers = cfg["cv"]["num_workers"]
    pin_memory  = cfg["cv"]["pin_memory"]

    tr_loader = DataLoader(ds_tr, batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True, prefetch_factor=8, )
    va_loader = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True, prefetch_factor=8, )
    te_loader = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True, prefetch_factor=8, )
    return tr_loader, va_loader, te_loader, used_seq_len



class SeqTensorDataset(Dataset):
    def __init__(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor, seq_len: int):
        assert X_tensor.device.type == "cpu" and y_tensor.device.type == "cpu"
        self.X = X_tensor.contiguous().float()   # [N,F]
        self.y = y_tensor.contiguous().long()    # [N]
        self.L = int(seq_len)
        self.idx = np.arange(self.L - 1, self.X.size(0) - 1)

    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.X[j - self.L + 1: j + 1], self.y[j]

def make_loaders_for_fold_from_pt(
    dt_index,                 # 時間索引，用於折分
    X_cpu: torch.Tensor,      # [N,F]
    y_cpu: torch.Tensor,      # [N]
    fold: dict,
    cfg: dict,
    selected_feats: list[str] | None = None,
    feat_cols: list[str] | None = None,
):
    # 1) 選特徵（若指定）
    if selected_feats is not None and feat_cols is not None:
        name2idx = {n:i for i,n in enumerate(feat_cols)}
        col_idx = [name2idx[n] for n in selected_feats if n in name2idx]
        if not col_idx:
            raise ValueError("selected_feats 與 feat_cols 不相符，無可用特徵。")
        X_cpu = X_cpu.index_select(1, torch.tensor(col_idx, dtype=torch.long))
    n_features = X_cpu.size(1)

    # 2) 序列長度
    seq_cfg = cfg["sequence"]["seq_len"]
    used_seq_len = int(np.median(seq_cfg)) if isinstance(seq_cfg, list) else int(seq_cfg)

    # 3) 以 fold 遮罩取連續區段
    idx_all = np.arange(len(dt_index))
    tv_idx = idx_all[fold["train_val_mask"]]
    ts_idx = idx_all[fold["test_mask"]]
    assert tv_idx.size > 0 and ts_idx.size > 0
    tv_start, tv_end = tv_idx[0], tv_idx[-1] + 1
    ts_start, ts_end = ts_idx[0], ts_idx[-1] + 1

    X_tv, y_tv = X_cpu[tv_start:tv_end], y_cpu[tv_start:tv_end]
    X_ts, y_ts = X_cpu[ts_start:ts_end], y_cpu[ts_start:ts_end]

    # 4) 奇數月內再切 train/val（連續切）
    split = int(len(X_tv) * cfg["cv"]["train_val_split"])
    X_tr, y_tr = X_tv[:split], y_tv[:split]
    X_va, y_va = X_tv[split:], y_tv[split:]
    for t in (X_tr, X_va, X_ts): assert t.size(0) > used_seq_len, "seq_len 太長導致空 dataset"

    # 5) Dataset / DataLoader
    ds_tr = SeqTensorDataset(X_tr, y_tr, used_seq_len)
    ds_va = SeqTensorDataset(X_va, y_va, used_seq_len)
    ds_te = SeqTensorDataset(X_ts, y_ts, used_seq_len)

    bs = int(cfg["train"]["batch_size"])
    nwrk = int(cfg["cv"]["num_workers"])
    pin  = bool(cfg["cv"]["pin_memory"])
    dl_kwargs = dict(batch_size=bs, num_workers=nwrk, pin_memory=pin)
    if nwrk > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=8))

    tr_loader = DataLoader(ds_tr, shuffle=True,  **dl_kwargs)
    va_loader = DataLoader(ds_va, shuffle=False, **dl_kwargs)
    te_loader = DataLoader(ds_te, shuffle=False, **dl_kwargs)
    return tr_loader, va_loader, te_loader, used_seq_len, n_features