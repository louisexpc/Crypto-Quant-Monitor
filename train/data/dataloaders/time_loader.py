def make_time_loaders_for_fold(df,
                               feat_cols: Optional[List[str]] = None,
                               target_col: Optional[str] = None,
                               fold: Dict = None,
                               cfg: Dict = None,
                               also_XGB: bool = False,
                               pre_feat_df: pd.DataFrame | None = None):
    """
    時間驅動（time-driven）資料載入器（precomputed-only）：
    - 僅從 cfg.features.precomputed.path 載入特徵；不做 runtime 特徵計算。
    - 以預算檔中的 OHLCV 產生 label（create_label）。
    - 依 fold 切出 train/val/test，執行縮放與清理，最後包成三個 DataLoader。
    - TimeSafeScaler：transform_full；sklearn 縮放器：fit on train，再 transform 其他 split。
    """

    task_type = cfg["task"]["type"]
    # 參考折疊的原始索引（由 objective.make_folds 基於此索引生成布林遮罩）
    ref_index = pd.DatetimeIndex(df.index)
    # 僅使用 precomputed 特徵
    pre_path = cfg["data"]["path"]
    if not pre_path and pre_feat_df is None:
        raise ValueError("請在 config.features.precomputed.path 指定預先計算的特徵檔 (.csv 或 .parquet)")
    if pre_feat_df is not None:
        feat_df = pre_feat_df.copy()
    else:
        p = str(pre_path)
        if p.endswith(".csv"):
            feat_df = pd.read_csv(p)
        elif p.endswith(".parquet"):
            feat_df = pd.read_parquet(p)
        else:
            raise ValueError("features.precomputed.path 只支援 .csv 或 .parquet")
        # set index from datetime/timestamp if present
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

        # 對齊時間網格並生成 label（保留 y_reg/y_cls 欄名）
        # 對齊時間網格：以 precomputed 索引生成完整網格
        full_idx = pd.date_range(feat_df.index.min(), feat_df.index.max(), freq=str(cfg["data"]["freq"]), tz="UTC")
        # 檢查預算檔是否包含 OHLCV 欄位
        ohlcv_cols = [c for c in ["open","high","low","close","volume"] if c in feat_df.columns]
        if len(ohlcv_cols) < 5:
            raise KeyError("預算特徵檔缺少 OHLCV 欄位（需要 open/high/low/close/volume）以產生 time-driven 標籤")
        dfb = feat_df.loc[:, ohlcv_cols].copy()
        dfb = dfb.reindex(full_idx)
        feat_df = feat_df.reindex(full_idx)

        # 篩選被 enable 的特徵（以 plan + OHLCV + 1-min）
        feat_cols = select_plan_columns(feat_df, cfg)
        drop_feat = [c for c in feat_df.columns if c not in set(feat_cols)]
        if drop_feat:
            print(f"[INFO] Dropping {len(drop_feat)} precomputed cols not enabled: {drop_feat[:10]}{' ...' if len(drop_feat)>10 else ''}")
        if not feat_cols:
            raise ValueError("計畫啟用的特徵在預算檔中皆不存在（feat_cols 為空），請檢查 plan 與預算欄位。")
        feat_df = feat_df.loc[:, feat_cols].astype(np.float32)

        is_reg = (task_type == "regression")
        y_series = create_label(dfb, cfg, return_what=("reg" if is_reg else "cls"))

        # 清理與對齊
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        valid_now = feat_df.notna().all(axis=1)
        valid_lbl = y_series.notna()
        keep = valid_now & valid_lbl
        feat_df = feat_df.loc[keep]
        y_series = y_series.loc[keep]

        # 時間區間篩選
        cv_start = pd.Timestamp(cfg["cv"]["start_date"]).tz_localize("UTC")
        cv_end   = pd.Timestamp(cfg["cv"]["end_date"]).tz_localize("UTC")
        cv_end   = cv_end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask_range = (feat_df.index >= cv_start) & (feat_df.index <= cv_end)
        feat_df = feat_df.loc[mask_range]
        y_series = y_series.loc[mask_range]

        # 構造 df 與 meta
        df = pd.concat([feat_df, y_series], axis=1)
        feat_cols = list(feat_df.columns)
        target_col = "y_reg" if is_reg else "y_cls"
    is_reg = (task_type == "regression")

    # Scaler（只 fit 在 train，否則會洩漏）
    scaler_kind = cfg["sequence"]["scaler"]
    scaler_window = cfg["sequence"]["seq_len"]
    min_frac = cfg["sequence"].get("min_frac", 0.2)

    # 1) 依 fold（基於 ref_index 的布林遮罩）映射到目前 df.index
    #    先將布林遮罩轉成時間集合，再以 isin 到當前 df.index 取得對應位置
    local_index = pd.DatetimeIndex(df.index)
    tv_times = ref_index[np.asarray(fold["train_val_mask"]).astype(bool)]
    te_times = ref_index[np.asarray(fold["test_mask"]).astype(bool)]
    tv_mask_local = local_index.isin(tv_times)
    te_mask_local = local_index.isin(te_times)
    df_tv_index = local_index[tv_mask_local]
    df_te_index = local_index[te_mask_local]

    split_ratio = cfg["cv"]["train_val_split"]
    split_pos = int(len(df_tv_index) * split_ratio)
    tr_idx = df_tv_index[:split_pos]
    va_idx = df_tv_index[split_pos:]
    te_idx = df_te_index

    # 1b) 決定要縮放的欄位（自動跳過 sign-like / 命名 pattern）
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


