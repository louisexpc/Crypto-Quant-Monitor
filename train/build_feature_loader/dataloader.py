# dataloader.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional, List, Dict


def split_fold_to_indices(df: pd.DataFrame, fold: Dict, cfg: Dict):
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
        # 統一使用 UTC（tz-aware）。若為 naive，視為 UTC。
        if getattr(dt_index, "tz", None) is None:
            dt_index = dt_index.tz_localize("UTC")
        else:
            dt_index = dt_index.tz_convert("UTC")
        self.dt_index = dt_index
        self.mode = mode
        # 用於 PeriodIndex 比較：Period.start_time 是 tz-naive，故這裡亦使用 tz-naive
        self.start_ts = pd.Timestamp(start_month)
        self.kwargs = kwargs

        # 2. 全部 folds 的月列表（但只保留起始時間之後的）
        # 先轉成 naive UTC 再做 to_period，避免 timezone 丟失警告
        dt_index_naive = self.dt_index.tz_convert("UTC").tz_localize(None)
        months_all = pd.PeriodIndex(dt_index_naive.to_period("M")).unique().sort_values()
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
        # 預先建立 naive 的 month period index，避免 tz 警告
        month_periods = pd.PeriodIndex(self.dt_index.tz_convert("UTC").tz_localize(None).to_period("M"))
        for m in self.months:
            if m.month % 2 == 1:  # 奇數月
                next_month = m + 1
                if next_month in self.months:
                    train_val_mask = (month_periods == m)
                    test_mask      = (month_periods == next_month)
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

        # 預先建立 naive 的 month period index，避免 tz 警告
        month_periods = pd.PeriodIndex(self.dt_index.tz_convert("UTC").tz_localize(None).to_period("M"))
        for m in test_months:
            # 該月的月初（naive）
            test_start = pd.Timestamp(m.start_time, tz="UTC")
            embargo_delta = pd.Timedelta(hours=embargo_hours)
            train_end = test_start - embargo_delta

            # 訓練（擴充式）：從 anchor 起累積到圖上 train_end_exclusive
            train_mask = (self.dt_index >= self.start_ts) & (self.dt_index < train_end)
            test_mask  = (month_periods == m)

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

        # 預先建立 naive 的 month period index，避免 tz 警告
        month_periods = pd.PeriodIndex(self.dt_index.tz_convert("UTC").tz_localize(None).to_period("M"))
        for m in test_months:
            i = self.months.index(m)
            if i < train_window or i + 1 >= len(self.months):
                continue

            train_start = pd.Timestamp(self.months[i - train_window].start_time, tz="UTC")
            test_start = pd.Timestamp(m.start_time, tz="UTC")
            embargo_delta = pd.Timedelta(hours=embargo_hours)
            train_end = test_start - embargo_delta

            train_mask = (self.dt_index >= train_start) & (self.dt_index < train_end)
            test_mask  = (month_periods == m)

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
from .event_dataset import EventDataset
from .indicators import IndicatorLibrary, FeatureComputer

def make_loaders_for_fold(df, feat_cols, target_col, fold, cfg, also_XGB: bool = False, pre_feat_df: pd.DataFrame | None = None):
    """
    依 fold 切出 train/val/test，執行縮放與清理，最後包成三個 DataLoader。
    主要差異點：
      - 若使用 TimeSafeScaler：先 transform_full，再針對每個 split 分別 dropna；
        並在 train split 裁掉 warm-up（scaler.warmup_len()）。
      - 若使用 sklearn 縮放器：先清理 train，再 fit_df(train)；val/test 僅 transform，不看未來。
    """

    # 事件驅動：改走 EventDataset 流程
    if str(cfg.get("label", {}).get("mode", "")).lower() == "event_tbm":
        return make_event_loaders_for_fold(df, feat_cols, fold, cfg, also_XGB=also_XGB, pre_feat_df=pre_feat_df)

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

    # Optional: print binary label distribution for standard classification
    if (not is_reg):
        try:
            def _vc(s):
                vc = s.value_counts().to_dict()
                vc.setdefault(0, 0); vc.setdefault(1, 0)
                return {int(k): int(v) for k, v in vc.items() if int(k) in (0,1)}
            lbl_tr = _vc(y_tr)
            lbl_va = _vc(y_va)
            lbl_te = _vc(y_te)
            tm = fold.get("test_month", str(""))
            print(f"[Fold] test_month={tm} | label_counts: TR={lbl_tr} VA={lbl_va} TE={lbl_te}")
            info["label_counts"] = {"train": lbl_tr, "val": lbl_va, "test": lbl_te}
        except Exception:
            pass

    return train_loader, val_loader, test_loader, info


def make_event_loaders_for_fold(df_events: pd.DataFrame,
                                feat_cols: List[str],
                                fold: Dict,
                                cfg: Dict,
                                also_XGB: bool = False,
                                pre_feat_df: pd.DataFrame | None = None):
    """
    Event-driven loader builder using EventDataset:
    - Recomputes features via FeatureComputer(plan) to get full 15m grid.
    - Applies scaler (time-safe rolling/ewm, or sklearn fit on train windows only).
    - Splits events by t0 alignment time using the given fold masks on df_events.index.
    """

    # === Split event index (t0 times) into train/val/test ===
    tr_idx, va_idx, te_idx = split_fold_to_indices(df_events, fold, cfg)

    # === Compute full-grid features via FeatureComputer (or reuse precomputed) ===
    if pre_feat_df is None:
        raw_path = cfg["data"]["path"]
        index_col = cfg["data"]["index_col"]
        freq = cfg["data"]["freq"]

        if str(raw_path).endswith(".csv"):
            df_raw = pd.read_csv(raw_path)
        elif str(raw_path).endswith(".parquet"):
            df_raw = pd.read_parquet(raw_path)
        else:
            raise ValueError("data.path must be .csv or .parquet")

        lib = IndicatorLibrary(df_raw, freq_check=freq, prefer_time_col=index_col)
        cache_dir = cfg["features"]["cache_dir"]
        worker_tag = os.environ.get("WORKER_TAG", "").strip()
        if worker_tag:
            cache_dir = os.path.join(cache_dir, worker_tag)
        fc = FeatureComputer(lib, cache_dir=cache_dir)
        plan = cfg["features"]["plan"]
        feat_df = fc.compute(plan, cfg)
    else:
        feat_df = pre_feat_df.copy()

    # Restrict to selected features only (ensure order)
    feat_df = feat_df.loc[:, [c for c in feat_cols if c in feat_df.columns]].astype(np.float32)

    # === Build scaler ===
    scaler_kind = cfg["sequence"]["scaler"]
    scaler_window = int(cfg["sequence"]["seq_len"])  # reuse seq_len
    min_frac = float(cfg["sequence"]["min_frac"]) if "min_frac" in cfg["sequence"] else 0.2
    scaler = _get_scaler(scaler_kind, window=scaler_window, min_frac=min_frac)

    # Helper: compute align positions for a set of t0 times
    def align_times(t0_index: pd.DatetimeIndex, idx_all: pd.DatetimeIndex, method: str) -> pd.DatetimeIndex:
        method = str(method).lower()
        t0u = pd.DatetimeIndex(t0_index)
        if t0u.tz is None:
            t0u = t0u.tz_localize("UTC")
        else:
            t0u = t0u.tz_convert("UTC")
        if method == "exact":
            pos = idx_all.get_indexer(t0u)
            valid = pos >= 0
            pos = pos[valid]
            return idx_all[pos]
        elif method == "pad":
            pos = idx_all.searchsorted(t0u, side="right") - 1
            valid = pos >= 0
            pos = pos[valid]
            return idx_all[pos]
        else:
            raise ValueError("align_method must be 'exact' or 'pad'")

    align_method = str(cfg.get("label", {}).get("align_method", "pad")).lower()
    L = int(cfg["sequence"]["seq_len"])  # window length
    idx_all = pd.DatetimeIndex(feat_df.index)

    # === Choose columns to scale using training windows only ===
    # Build union of all bars used in train-event windows: [p-L, p)
    train_align = align_times(tr_idx, idx_all, align_method)
    fit_pos = []
    # map times to integer positions
    pos_map = pd.Series(np.arange(len(idx_all)), index=idx_all)
    for at in train_align:
        p = int(pos_map.get(at, -1))
        if p <= 0:
            continue
        start = max(0, p - L)
        fit_pos.extend(range(start, p))
    fit_pos = np.unique(np.array(fit_pos, dtype=int))
    fit_index = idx_all[fit_pos] if len(fit_pos) else train_align  # fallback: use align times

    cols_to_scale = pick_cols_to_scale(feat_df.loc[fit_index, feat_cols], feat_cols)

    # === Apply scaler ===
    if hasattr(scaler, "is_timesafe") and scaler.is_timesafe:
        feat_scaled = scaler.transform_full(feat_df, cols_to_scale=cols_to_scale)
    else:
        if scaler is None:
            feat_scaled = feat_df
            sklearn_scaler = None
        else:
            sklearn_scaler = ColumnSubsetScaler(scaler, all_cols=feat_cols, cols_to_scale=cols_to_scale)
            sklearn_scaler.fit_df(feat_df.loc[fit_index, feat_cols])
            arr = feat_df.loc[:, feat_cols].values.astype(np.float32, copy=False)
            arr = sklearn_scaler.transform(arr)
            feat_scaled = feat_df.copy()
            feat_scaled.loc[:, feat_cols] = arr

    # === Build three EventDataset partitions (preloaded on device) ===
    tbm_csv_path = cfg["label"]["tbm_csv_path"]
    keep_sides = str(cfg["label"].get("keep_sides", "both")).lower()
    runtime_device = cfg["device"]
    bs = int(cfg["train"]["batch_size"])

    tr_align = align_times(tr_idx, idx_all, align_method)
    va_align = align_times(va_idx, idx_all, align_method)
    te_align = align_times(te_idx, idx_all, align_method)

    ds_tr = EventDataset(feat_scaled, tbm_csv_path, seq_len=L,
                         feature_cols=feat_cols, keep_sides=keep_sides,
                         align_method=align_method, device=runtime_device,
                         allowed_align_index=tr_align)
    ds_va = EventDataset(feat_scaled, tbm_csv_path, seq_len=L,
                         feature_cols=feat_cols, keep_sides=keep_sides,
                         align_method=align_method, device=runtime_device,
                         allowed_align_index=va_align)
    ds_te = EventDataset(feat_scaled, tbm_csv_path, seq_len=L,
                         feature_cols=feat_cols, keep_sides=keep_sides,
                         align_method=align_method, device=runtime_device,
                         allowed_align_index=te_align)

    # DataLoader（資料已在目標裝置，故不需 pin_memory）
    train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(ds_te, batch_size=bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)

    # Label distribution per split (0/1 counts)
    def _counts(ds):
        import torch
        y = ds.y
        if isinstance(y, torch.Tensor):
            y = y.detach().to("cpu")
            uniq, cnt = torch.unique(y, return_counts=True)
            d = {int(u.item()): int(c.item()) for u, c in zip(uniq, cnt)}
        else:
            import numpy as np
            arr = np.asarray(y)
            u, c = np.unique(arr, return_counts=True)
            d = {int(uu): int(cc) for uu, cc in zip(u, c)}
        # Ensure keys 0/1 exist
        d.setdefault(0, 0); d.setdefault(1, 0)
        return d

    lbl_tr, lbl_va, lbl_te = _counts(ds_tr), _counts(ds_va), _counts(ds_te)
    tm = fold.get("test_month", str(""))
    print(f"[EventFold] test_month={tm} | label_counts: TR={lbl_tr} VA={lbl_va} TE={lbl_te}")

    info = {"feat_cols": feat_cols, "target_col": "label", "label_counts": {"train": lbl_tr, "val": lbl_va, "test": lbl_te}}

    # XGB pack（flatten sequences；僅為滿足流程，分類時通常不使用）
    if also_XGB:
        def as_np(x):
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().to("cpu").numpy()
            return np.asarray(x)
        Xtr = as_np(ds_tr.X).reshape(len(ds_tr), -1).astype(np.float32, copy=False)
        Xva = as_np(ds_va.X).reshape(len(ds_va), -1).astype(np.float32, copy=False)
        Xte = as_np(ds_te.X).reshape(len(ds_te), -1).astype(np.float32, copy=False)
        ytr = as_np(ds_tr.y).astype(np.int64, copy=False)
        yva = as_np(ds_va.y).astype(np.int64, copy=False)
        yte = as_np(ds_te.y).astype(np.int64, copy=False)
        info["XGB"] = {
            "X_tr": Xtr, "y_tr": ytr,
            "X_va": Xva, "y_va": yva,
            "X_te": Xte, "y_te": yte,
            "scaler": None,
            "cols_to_scale": [],
        }

    return train_loader, val_loader, test_loader, info
