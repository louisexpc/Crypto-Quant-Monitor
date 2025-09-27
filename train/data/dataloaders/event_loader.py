def make_event_loaders_for_fold(df_events: pd.DataFrame,
                                feat_cols: List[str],
                                fold: Dict,
                                cfg: Dict,
                                also_XGB: bool = False,
                                pre_feat_df: pd.DataFrame | None = None):
    """
    Event-driven loader builder using EventDataset:
    - Loads precomputed features (15m grid) from cfg.features.precomputed.path.
    - Applies scaler (time-safe rolling/ewm, or sklearn fit on train windows only).
    - Splits events by t0 alignment time using the given fold masks on df_events.index.
    """

    # === Split event index (t0 times) into train/val/test ===
    tr_idx, va_idx, te_idx = split_fold_to_indices(df_events, fold, cfg)

    # === Load full-grid features from precomputed file only ===
    if pre_feat_df is not None:
        feat_df = pre_feat_df.copy()
    else:
        pre_path = cfg["data"]["path"]
        if not pre_path:
            raise ValueError("event 模式需要 config.features.precomputed.path 指定預算特徵檔")
        p = str(pre_path)
        if p.endswith(".csv"):
            feat_df = pd.read_csv(p)
        elif p.endswith(".parquet"):
            feat_df = pd.read_parquet(p)
        else:
            raise ValueError("features.precomputed.path 只支援 .csv 或 .parquet")
        if "datetime" in feat_df.columns:
            idx = pd.to_datetime(feat_df["datetime"], errors="coerce", utc=True)
            feat_df = feat_df.drop(columns=["datetime"]) 
            feat_df.index = idx
        elif "timestamp" in feat_df.columns:
            ts = pd.to_numeric(feat_df["timestamp"], errors="coerce").astype("Int64")
            unit = "ms" if (ts.dropna().iloc[0] if len(ts.dropna()) else 0) > 1_000_000_000_000 else "s"
            idx = pd.to_datetime(ts, unit=unit, utc=True)
            feat_df = feat_df.drop(columns=["timestamp"]) 
            feat_df.index = idx
        feat_df = feat_df.sort_index()
        feat_df = feat_df[~feat_df.index.duplicated(keep="last")]

    # Restrict to selected features only (ensure order)
    if not feat_cols:
        feat_cols = select_plan_columns(feat_df, cfg)
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
    # pos_map = pd.Series(np.arange(len(idx_all)), index=idx_all)
    # for at in train_align:
    #     p = int(pos_map.get(at, -1))
    #     if p <= 0:
    #         continue
    #     start = max(0, p - L)
    #     fit_pos.extend(range(start, p))
    # fit_pos = np.unique(np.array(fit_pos, dtype=int))
    # fit_index = idx_all[fit_pos] if len(fit_pos) else train_align  # fallback: use align times

    # vectorized build of fit positions: union over [p-L, p)
    p_vec = idx_all.searchsorted(train_align, side="right") - 1  # [N_tr]
    p_vec = p_vec[p_vec >= 0]
    if len(p_vec):
        rng = np.arange(L, dtype=np.int32)
        fit_pos = (p_vec[:, None] - rng[None, :]).reshape(-1)
        fit_pos = fit_pos[(fit_pos >= 0) & (fit_pos < len(idx_all))]
        fit_pos = np.unique(fit_pos)
        fit_index = idx_all[fit_pos]
    else:
        fit_index = train_align

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
