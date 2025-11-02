# train/data/dataloaders/time_loader.py
"""
Time-driven DataLoader builder.

把時間驅動 (time-driven) 的 bar 級資料，對齊到離線預算好的特徵格點，依折疊切成
train/val/test，套用時間安全或 sklearn 縮放器，最後回傳三個 DataLoader 與 info。
"""
from __future__ import annotations
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from train.data.dataloaders.base import (
    load_precomputed_features,
    reindex_to_full_grid,
    apply_scaling,
)
from train.data.labeling import create_label
from train.data.dataset.time_dataset import SeqDataset


def make_time_loaders_for_fold(
    df: pd.DataFrame,
    feat_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    fold: Dict | None = None,
    cfg: Dict | None = None,
    also_XGB: bool = False,
):
    """
    時間驅動 DataLoader：僅用預算特徵、依 plan 篩欄、算 OHLCV label、縮放、切 TR/VA/TE。
    """
    assert cfg is not None and fold is not None, "cfg 與 fold 皆不可為 None"
    task_type = cfg["task"]["type"]
    ref_index = pd.DatetimeIndex(df.index)

    # 1) 取得特徵與標籤來源
    feat_df = load_precomputed_features(path=cfg["data"]["path"]).astype(np.float32)
    feat_cols = [c for c in feat_df.columns if np.issubdtype(feat_df[c].dtype, np.number)]

    freq = (cfg.get("data", {}) or {}).get("freq")
    if freq and not feat_df.empty:
        feat_df = reindex_to_full_grid(feat_df, str(freq))

    # 2) 先抓 OHLCV 做 label（若快取沒有時間標籤，現場計算）
    need = ["open","high","low","close","volume"]
    miss = [c for c in need if c not in feat_df.columns]
    if miss:
        raise KeyError(f"預算特徵檔缺少 OHLCV 欄位: {miss}")
    dfb = feat_df.loc[:, need].copy()


    # 4) 產生 y（若快取有現成資料則使用）
    is_reg = (task_type == "regression")
    y_series = create_label(dfb, cfg, return_what=("reg" if is_reg else "cls"))
    y_series = y_series.reindex(feat_df.index)

    # 5) 清理 + CV 範圍 + 與 fold 對齊（沿用你現有邏輯）
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    keep = feat_df.notna().all(axis=1) & y_series.notna()
    feat_df = feat_df.loc[keep]; y_series = y_series.loc[keep]

    cv_start = pd.Timestamp(cfg["cv"]["start_date"]).tz_localize("UTC")
    cv_end   = (pd.Timestamp(cfg["cv"]["end_date"]).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1))
    mask = (feat_df.index >= cv_start) & (feat_df.index <= cv_end)
    feat_df = feat_df.loc[mask]; y_series = y_series.loc[mask]

    work = pd.concat([feat_df, y_series], axis=1)
    feat_cols = list(feat_df.columns)
    target_col = target_col or ("y_reg" if is_reg else "y_cls")

    # 6) fold 對齊
    local_index = pd.DatetimeIndex(work.index)
    tv_times = ref_index[np.asarray(fold["train_val_mask"]).astype(bool)]
    te_times = ref_index[np.asarray(fold["test_mask"]).astype(bool)]
    tv_mask_local = local_index.isin(tv_times)
    te_mask_local = local_index.isin(te_times)
    df_tv_index = local_index[tv_mask_local]
    df_te_index = local_index[te_mask_local]
    split_pos = int(len(df_tv_index) * cfg["cv"]["train_val_split"])
    tr_idx, va_idx, te_idx = df_tv_index[:split_pos], df_tv_index[split_pos:], df_te_index

    # 7) 縮放（time-safe 直接全段；sklearn 用 train bar 作 fit_index）
    L = int(cfg["sequence"]["seq_len"])
    min_frac = float(cfg["sequence"].get("min_frac", 0.2))
    scaled, sk_scaler, cols_to_scale = apply_scaling(
        df=work, feat_cols=feat_cols, scaler_kind=cfg["sequence"]["scaler"],
        L=L, min_frac=min_frac, fit_index=tr_idx
    )

    # 8) split → Dataset / Loader（保持你原本的設定）
    def _clean(X_df, y_s):
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        valid = X_df.notna().all(axis=1) & y_s.notna()
        return X_df.loc[valid], y_s.loc[valid]

    X_tr, y_tr = _clean(scaled.loc[tr_idx, feat_cols], scaled.loc[tr_idx, target_col])
    X_va, y_va = _clean(scaled.loc[va_idx, feat_cols], scaled.loc[va_idx, target_col])
    X_te, y_te = _clean(scaled.loc[te_idx, feat_cols], scaled.loc[te_idx, target_col])

    runtime_device = cfg["device"]; bs = int(cfg["train"]["batch_size"])
    preload = bool(cfg.get("sequence", {}).get("preload_to_gpu", False))
    ds_device = runtime_device if (preload and runtime_device == "cuda") else "cpu"
    stride = cfg["sequence"]["stride"]; anchor = int(cfg["sequence"]["stride_anchor"]) % stride
    label_dtype = "float" if is_reg else "long"

    ds_tr = SeqDataset(X_tr, y_tr, L, scaler=sk_scaler, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)
    ds_va = SeqDataset(X_va, y_va, L, scaler=sk_scaler, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)
    ds_te = SeqDataset(X_te, y_te, L, scaler=sk_scaler, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)

    pin = (ds_device == "cpu" and runtime_device == "cuda"); nw = 10 if pin else 0
    train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=pin)
    test_loader  = DataLoader(ds_te, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=pin)

    info = {"feat_cols": feat_cols, "target_col": target_col}
    if also_XGB:
        Xtr = X_tr.values.astype(np.float32, copy=False)
        Xva = X_va.values.astype(np.float32, copy=False)
        Xte = X_te.values.astype(np.float32, copy=False)
        if sk_scaler is not None:
            Xtr = sk_scaler.transform(Xtr); Xva = sk_scaler.transform(Xva); Xte = sk_scaler.transform(Xte)
        y_dtype = np.float32 if is_reg else np.int64
        info["XGB"] = {"X_tr": Xtr, "y_tr": y_tr.values.astype(y_dtype, copy=False),
                       "X_va": Xva, "y_va": y_va.values.astype(y_dtype, copy=False),
                       "X_te": Xte, "y_te": y_te.values.astype(y_dtype, copy=False),
                       "scaler": sk_scaler, "cols_to_scale": cols_to_scale}

    return train_loader, val_loader, test_loader, info
