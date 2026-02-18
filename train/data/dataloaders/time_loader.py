# train/data/dataloaders/time_loader.py
"""
Time-driven DataLoader builder.

把時間驅動 (time-driven) 的 bar 級資料，對齊到離線預算好的特徵格點，依折疊切成
train/val/test，最後回傳三個 DataLoader 與 info。
"""
from __future__ import annotations
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from train.data.dataloaders.base import (
    load_precomputed_features,
    load_ohlcv_for_label,
    reindex_to_full_grid,
    flatten_micro_features,
    resolve_timezone_name,
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
    時間驅動 DataLoader：僅用預算特徵、依 plan 篩欄、算 OHLCV label、切 TR/VA/TE。
    """
    assert cfg is not None and fold is not None, "cfg 與 fold 皆不可為 None"
    task_type = cfg["task"]["type"]
    ref_index = pd.DatetimeIndex(df.index)
    data_cfg = (cfg.get("data", {}) or {})

    feat_path = data_cfg.get("feat_path")
    if not feat_path:
        raise KeyError("cfg.data.feat_path is required.")
    ohlcv_path = data_cfg.get("ohlcv_fng_path")
    if not ohlcv_path:
        raise KeyError("cfg.data.ohlcv_fng_path is required.")

    data_tz = resolve_timezone_name(data_cfg.get("time_zone"), default="Asia/Taipei")

    def _to_tz(ts_like):
        """Convert timestamp-like object to configured data timezone.

        Args:
            ts_like: Input timestamp-like value.
        Returns:
            Timezone-aware pandas Timestamp in data_tz.
        """
        ts = pd.Timestamp(ts_like)
        if ts.tzinfo is None:
            return ts.tz_localize(data_tz)
        return ts.tz_convert(data_tz)

    cv_start = _to_tz(cfg["cv"]["start_date"])
    cv_end_cfg = _to_tz(cfg["cv"]["end_date"])
    post_cfg = (cfg.get("post_infer", {}) or {})
    post_end = post_cfg.get("date_end")
    ts_end_limit = _to_tz(post_end) if post_end else cv_end_cfg

    # 1) 取得特徵與標籤來源
    feat_df = load_precomputed_features(
        path=feat_path,
        drop_ohlcv=False,
        target_tz=data_tz,
    ).astype(np.float32)
    ohlcv_df = load_ohlcv_for_label(
        path=ohlcv_path,
        target_tz=data_tz,
    ).astype(np.float32)

    micro_cfg = data_cfg.get("micro", {}) or {}
    micro_df = None
    if micro_cfg.get("enabled") and micro_cfg.get("path"):
        micro_df = load_precomputed_features(
            path=micro_cfg["path"],
            target_tz=data_tz,
        ).astype(np.float32)

    # 日期裁剪：cv.start_date ~ min(cv/post_infer end)
    ts_end_candidates = [
        ts_end_limit,
        pd.DatetimeIndex(feat_df.index).max(),
        pd.DatetimeIndex(ohlcv_df.index).max(),
    ]
    if micro_df is not None and len(micro_df.index):
        ts_end_candidates.append(pd.DatetimeIndex(micro_df.index).max())
    ts_end = min(ts_end_candidates)

    feat_df = feat_df.loc[feat_df.index <= ts_end]
    if feat_df.isna().any().any():
        raise ValueError("[time_loader] 15m features contain NaN/Inf; please sanitize upstream.")

    if micro_df is not None:
        window_len = int(micro_cfg.get("window_len", 15))
        feat_df = flatten_micro_features(
            feat_df=feat_df,
            micro_df=micro_df,
            cv_start=cv_start,
            ts_end=ts_end,
            window_len=window_len,
        )

    feat_df = feat_df.loc[(feat_df.index >= cv_start) & (feat_df.index <= ts_end)]

    freq = (cfg.get("data", {}) or {}).get("freq")
    if freq and not feat_df.empty:
        feat_df = reindex_to_full_grid(feat_df, str(freq))

    # 2) 先抓 OHLCV 做 label（若快取沒有時間標籤，現場計算）
    dfb = ohlcv_df.reindex(feat_df.index)
    missing_mask = dfb.isna().any(axis=1)
    if missing_mask.any():
        miss_idx = dfb.index[missing_mask][:5].tolist()
        raise ValueError(f"[time_loader] ohlcv_fng_path 無法完整對齊特徵索引，缺值樣本例：{miss_idx}")

    # 檢查 NaN
    nan_rows = feat_df.isna().any(axis=1)
    if nan_rows.any():
        raise ValueError(f"[time_loader] 預算特徵尚含 NaN/Inf，請調整特徵匯出流程；例：{feat_df.index[nan_rows][:5].tolist()}")

    # 4) 產生 y（若快取有現成資料則使用）
    is_reg = (task_type == "regression")
    y_series = create_label(dfb, cfg, return_what=("reg" if is_reg else "cls"))
    y_series = y_series.reindex(feat_df.index)
    if y_series.isna().any():
        raise ValueError("[time_loader] 產生標籤後仍含 NaN，請調整標籤設定或資料。")

    # 5) CV 範圍 + 與 fold 對齊（沿用你現有邏輯）
    mask = (feat_df.index >= cv_start) & (feat_df.index <= ts_end)
    feat_df = feat_df.loc[mask]; y_series = y_series.loc[mask]

    work = pd.concat([feat_df, y_series], axis=1)
    feat_cols = [c for c in feat_df.columns if np.issubdtype(feat_df[c].dtype, np.number)]
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

    # 7) 特徵已在上游處理，這裡不再做任何縮放
    L = int(cfg["sequence"]["seq_len"])
    scaled = work

    # 8) split → Dataset / Loader（保持你原本的設定）
    def _clean(X_df, y_s):
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        valid = X_df.notna().all(axis=1) & y_s.notna()
        return X_df.loc[valid], y_s.loc[valid]

    X_tr, y_tr = _clean(scaled.loc[tr_idx, feat_cols], scaled.loc[tr_idx, target_col])
    X_va, y_va = _clean(scaled.loc[va_idx, feat_cols], scaled.loc[va_idx, target_col])
    X_te, y_te = _clean(scaled.loc[te_idx, feat_cols], scaled.loc[te_idx, target_col])

    runtime_device = str(cfg["device"]); bs = int(cfg["train"]["batch_size"])
    runtime_is_cuda = runtime_device.startswith("cuda")
    preload_to_gpu = bool(data_cfg.get("preload_to_gpu", True))
    ds_device = runtime_device if (runtime_is_cuda and preload_to_gpu) else "cpu"
    stride_cfg = cfg["sequence"].get("stride", 1)
    if isinstance(stride_cfg, list):
        if len(stride_cfg) != 1:
            raise ValueError("[time_loader] sequence.stride list 僅允許單一值。")
        stride = int(stride_cfg[0])
    else:
        stride = int(stride_cfg)
    if stride <= 0:
        raise ValueError("[time_loader] sequence.stride 必須 > 0。")
    anchor = 0
    label_dtype = "float" if is_reg else "long"

    ds_tr = SeqDataset(X_tr, y_tr, L, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)
    ds_va = SeqDataset(X_va, y_va, L, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)
    ds_te = SeqDataset(X_te, y_te, L, device=ds_device, label_dtype=label_dtype, stride=stride, anchor=anchor)

    pin = (ds_device == "cpu" and runtime_is_cuda); nw = 10 if pin else 0
    train_loader = DataLoader(ds_tr, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=pin)
    test_loader  = DataLoader(ds_te, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=pin)

    info = {"feat_cols": feat_cols, "target_col": target_col}
    if also_XGB:
        Xtr = X_tr.values.astype(np.float32, copy=False)
        Xva = X_va.values.astype(np.float32, copy=False)
        Xte = X_te.values.astype(np.float32, copy=False)
        y_dtype = np.float32 if is_reg else np.int64
        info["XGB"] = {"X_tr": Xtr, "y_tr": y_tr.values.astype(y_dtype, copy=False),
                       "X_va": Xva, "y_va": y_va.values.astype(y_dtype, copy=False),
                       "X_te": Xte, "y_te": y_te.values.astype(y_dtype, copy=False)}

    return train_loader, val_loader, test_loader, info
