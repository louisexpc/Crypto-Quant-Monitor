# train/data/dataloaders/event_loader.py
from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from train.data.dataloaders.base import (
    load_precomputed_features,
    align_times,
    label_counts_from_ds,
    build_loaders,
    flatten_micro_features,
    resolve_timezone_name,
)
from train.data.folds import split_fold_to_indices
from train.data.dataset.event_dataset import EventDataset


def make_event_loaders_for_fold(
    df_events: pd.DataFrame,
    feat_cols: Optional[List[str]] = None,
    fold: Dict | None = None,
    cfg: Dict | None = None,
    also_XGB: bool = False,
    return_flattened: bool = False,
):
    """
    事件驅動 DataLoader：對齊 t0→特徵格點，依 fold 切 TR/VA/TE，直接包成 EventDataset。
    """
    assert cfg is not None and fold is not None, "cfg 與 fold 皆不可為 None"
    # 1) split
    tr_idx, va_idx, te_idx = split_fold_to_indices(df_events, fold, cfg)
    data_cfg = (cfg.get("data", {}) or {})

    feat_path = data_cfg.get("feat_path")
    if not feat_path:
        raise KeyError("cfg.data.feat_path is required.")

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

    # 2) 讀特徵 + 過濾欄位
    feat_df = load_precomputed_features(
        path=feat_path,
        target_tz=data_tz,
    )

    micro_cfg = data_cfg.get("micro", {}) or {}
    micro_df = None
    if micro_cfg.get("enabled") and micro_cfg.get("path"):
        micro_df = load_precomputed_features(
            path=micro_cfg["path"],
            target_tz=data_tz,
        )

    ts_end_candidates = [ts_end_limit, pd.DatetimeIndex(feat_df.index).max()]
    if micro_df is not None and len(micro_df.index):
        ts_end_candidates.append(pd.DatetimeIndex(micro_df.index).max())
    ts_end = min(ts_end_candidates)

    if micro_df is not None:
        window_len = int(micro_cfg.get("window_len", 15))
        feat_df = flatten_micro_features(
            feat_df=feat_df,
            micro_df=micro_df,
            cv_start=cv_start,
            ts_end=ts_end,
            window_len=window_len,
        )

    # 最終裁剪到 [cv_start, ts_end]
    feat_df = feat_df.loc[(feat_df.index >= cv_start) & (feat_df.index <= ts_end)]
    if feat_df.isna().any().any():
        raise ValueError("[event_loader] 預算特徵尚含 NaN/Inf；請在匯出/展平階段清理。")

    feat_cols = [c for c in feat_df.columns if np.issubdtype(feat_df[c].dtype, np.number)]

    # 3) 對齊 train/val/test 事件到特徵格點
    L = int(cfg["sequence"]["seq_len"])
    idx_all = pd.DatetimeIndex(feat_df.index)
    align_method = str(cfg.get("label", {}).get("align_method", "pad")).lower()
    tr_align = align_times(tr_idx, idx_all, align_method)
    va_align = align_times(va_idx, idx_all, align_method)
    te_align = align_times(te_idx, idx_all, align_method)

    # 4) 建三個 EventDataset
    tbm_csv_path = cfg["label"]["tbm_csv_path"]
    keep_sides = str(cfg["label"].get("keep_sides", "both")).lower()
    runtime_device = str(cfg["device"])
    runtime_is_cuda = runtime_device.startswith("cuda")
    preload_to_gpu = bool(data_cfg.get("preload_to_gpu", True))
    ds_device = runtime_device if (runtime_is_cuda and preload_to_gpu) else "cpu"
    bs = int(cfg["train"]["batch_size"])

    ds_tr = EventDataset(feat_df, tbm_csv_path, seq_len=L, feature_cols=feat_cols,
                         keep_sides=keep_sides, align_method=align_method,
                         device=ds_device, allowed_align_index=tr_align)
    ds_va = EventDataset(feat_df, tbm_csv_path, seq_len=L, feature_cols=feat_cols,
                         keep_sides=keep_sides, align_method=align_method,
                         device=ds_device, allowed_align_index=va_align)
    ds_te = EventDataset(feat_df, tbm_csv_path, seq_len=L, feature_cols=feat_cols,
                         keep_sides=keep_sides, align_method=align_method,
                         device=ds_device, allowed_align_index=te_align)

    # 5) DataLoader + label 分佈
    train_loader, val_loader, test_loader = build_loaders(ds_tr, ds_va, ds_te, batch_size=bs, device=runtime_device)
    lbl_tr, lbl_va, lbl_te = label_counts_from_ds(ds_tr), label_counts_from_ds(ds_va), label_counts_from_ds(ds_te)
    print(f"[EventFold] test_month={fold.get('test_month','')} | label_counts: TR={lbl_tr} VA={lbl_va} TE={lbl_te}")

    info = {"feat_cols": feat_cols, "target_col": "label",
            "label_counts": {"train": lbl_tr, "val": lbl_va, "test": lbl_te}}

    # 6) XGB（可選）
    if also_XGB:
        def as_np(x):
            import torch
            return x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        Xtr = as_np(ds_tr.X).reshape(len(ds_tr), -1).astype(np.float32, copy=False)
        Xva = as_np(ds_va.X).reshape(len(ds_va), -1).astype(np.float32, copy=False)
        Xte = as_np(ds_te.X).reshape(len(ds_te), -1).astype(np.float32, copy=False)
        ytr = as_np(ds_tr.y).astype(np.int64, copy=False)
        yva = as_np(ds_va.y).astype(np.int64, copy=False)
        yte = as_np(ds_te.y).astype(np.int64, copy=False)
        info["XGB"] = {"X_tr": Xtr, "y_tr": ytr, "X_va": Xva, "y_va": yva, "X_te": Xte, "y_te": yte}

    if return_flattened:
        return train_loader, val_loader, test_loader, info, feat_df
    return train_loader, val_loader, test_loader, info







"""
test case
"""
from pathlib import Path
if __name__ == "__main__":
    # Quick manual test: flatten 1m trades into m0~m14_* columns via make_event_loaders_for_fold.
    import argparse

    parser = argparse.ArgumentParser(description="Debug make_event_loaders_for_fold flattening.")
    parser.add_argument("--config", type=str, default="train/config.yaml", help="YAML config path.")
    args = parser.parse_args()

    import yaml
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # dummy events/fold covering all rows
    data_cfg = (cfg.get("data", {}) or {})
    data_tz = resolve_timezone_name(data_cfg.get("time_zone"), default="Asia/Taipei")
    feat_df_all = load_precomputed_features(path=cfg["data"]["feat_path"], target_tz=data_tz)
    idx_all = pd.DatetimeIndex(feat_df_all.index)
    mask = np.ones(len(idx_all), dtype=bool)
    fold = {"train_val_mask": mask, "test_mask": mask.copy()}

    train_loader, val_loader, test_loader, info, flat_df = make_event_loaders_for_fold(
        df_events=feat_df_all,
        fold=fold,
        cfg=cfg,
        also_XGB=False,
        return_flattened=True,
    )
    print(flat_df.head())
    # out.to_csv("data/precomputed/btcusdt_15m_features/output.csv", index=False, encoding="utf-8")
