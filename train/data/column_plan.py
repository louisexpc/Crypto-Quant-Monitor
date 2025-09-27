def select_plan_columns(feat_df: pd.DataFrame, cfg: Dict) -> List[str]:
    """
    根據 cfg.features.plan 產生要使用的欄位集合，包含：
    - OHLCV（僅在 plan 中對應項目 enabled 時才保留）
    - 所有 1-min 欄位（以 DEFAULT_MINUTE_PREFIXES 為前綴，例如 'm_...'; 請確保已於上游 drop 掉 datetime/timestamp）
    - 由 plan 中 enabled 的特徵映射出的實際欄位（與 feat_df.columns 取交集）

    輸出順序遵守原始 feat_df.columns 的順序。
    """
    from train.data.features.indicators import FeatureComputer, DEFAULT_MINUTE_PREFIXES
    import re

    cols_all = list(map(str, feat_df.columns))
    # OHLCV 按 plan 控制是否納入特徵（label 計算已在外層處理，不受此處影響）
    ohlcv_keep: set[str] = set()

    # Minute (1-min flattened) selection via config.features.min_trade_feat
    # Expect column form like: m_[-lag]_<base>
    min_feat_list = list(((cfg.get("features", {}) or {}).get("min_trade_feat", [])) or [])
    minute_cols: set[str] = set()
    if min_feat_list:
        m_pat = re.compile(r"^m_(-?\d+)_(.+)$")
        for c in cols_all:
            if not any(str(c).startswith(p) for p in DEFAULT_MINUTE_PREFIXES):
                continue
            m = m_pat.match(str(c))
            if not m:
                continue
            base = m.group(2)
            if base in min_feat_list:
                minute_cols.add(c)

    plan = (cfg.get("features", {}) or {}).get("plan", {}) or {}
    try:
        specs = FeatureComputer._enabled_features(plan)
    except Exception:
        specs = []

    want = set()

    def add_if_present(names):
        if isinstance(names, str):
            if names in feat_df.columns:
                want.add(names)
        else:
            for n in names:
                if n in feat_df.columns:
                    want.add(n)

    for item in specs:
        name = str(item.get("name", "")).upper()
        kw = item.get("kwargs", {}) or {}

        if name == "SMA":
            L = int(kw.get("length", 0)); s = str(kw.get("sign", "false")).lower()
            if s in ("true","sign","both"): add_if_present(f"SSMA_{L}")
            if s in ("false","cont","both"): add_if_present(f"SMA_{L}")
        elif name == "EMA":
            L = int(kw.get("length", 0)); s = str(kw.get("sign", "false")).lower()
            if s in ("true","sign","both"): add_if_present(f"SEMA_{L}")
            if s in ("false","cont","both"): add_if_present(f"EMA_{L}")
        elif name == "TEMA":
            L = int(kw.get("length", 0)); s = str(kw.get("sign", "false")).lower()
            if s in ("true","sign","both"): add_if_present(f"STEMA_{L}")
            if s in ("false","cont","both"): add_if_present(f"TEMA_{L}")
        elif name == "MACD":
            f = int(kw.get("fast", 12)); sl = int(kw.get("slow", 26)); sg = int(kw.get("signal", 9))
            s = str(kw.get("sign", "false")).lower()
            if s in ("true","sign","both"): add_if_present(f"SMACD_{f}_{sl}_{sg}")
            if s in ("false","cont","both"): add_if_present(f"MACD_{f}_{sl}_{sg}")
        elif name == "SLOPE":
            L = int(kw.get("length", 0)); add_if_present(f"SLOPE_{L}")
        elif name == "TTM_TRND":
            L = int(kw.get("length", 6)); add_if_present(f"TTM_TRND_{L}")
        elif name == "DPO":
            L = int(kw.get("length", 0)); add_if_present(f"DPO_{L}")
        elif name == "AMATE_LR":
            f = int(kw.get("fast", 8)); sl = int(kw.get("slow", 21)); m = int(kw.get("mamode", 2))
            add_if_present(f"AMATe_LR_{f}_{sl}_{m}")

        elif name == "RSI":
            L = int(kw.get("length", 14)); add_if_present(f"RSI_{L}")
        elif name == "MOM":
            L = int(kw.get("length", 30)); add_if_present(f"MOM_{L}")
        elif name == "STOCH":
            k = int(kw.get("k", 14)); add_if_present([f"STOCHk_{k}", f"STOCHd_{k}"])
        elif name == "KDJ":
            k = int(kw.get("k", 9)); d = int(kw.get("d", 3)); add_if_present(f"J_{k}_{d}")
        elif name == "UO":
            f = int(kw.get("fast", 7)); md = int(kw.get("medium", 14)); sl = int(kw.get("slow", 28)); add_if_present(f"UO_{f}_{md}_{sl}")
        elif name == "RVI":
            L = int(kw.get("length", 14)); add_if_present(f"RVI_{L}")
        elif name == "CCI":
            L = int(kw.get("length", 14)); c = float(kw.get("c", 0.015)); add_if_present(f"CCI_{L}_{c}")
        elif name == "ZS":
            L = int(kw.get("length", 30)); add_if_present(f"ZS_{L}")
        elif name == "WILLR":
            L = int(kw.get("length", 14)); add_if_present(f"WILLR_{L}")

        elif name == "TRUERANGE":
            add_if_present("TRUERANGE_1")
        elif name == "RANGE":
            W = int(kw.get("window", 24)); add_if_present(f"RANGE_{W}")
        elif name == "ATR":
            L = int(kw.get("length", 14)); add_if_present(f"ATR_{L}")
            if bool(kw.get("pct", True)): add_if_present(f"ATRP_{L}")
        elif name == "MASSI":
            f = int(kw.get("fast", 9)); sl = int(kw.get("slow", 25)); add_if_present(f"MASSI_{f}_{sl}")
        elif name == "BBP":
            L = int(kw.get("length", 5)); st = float(kw.get("std", 2.0)); add_if_present(f"BBP_{L}_{st}")
        elif name == "EWMRET":
            hls = kw.get("halflife", [])
            if isinstance(hls, int): hls = [hls]
            for hl in hls:
                add_if_present([f"EWM_M_{int(hl)}", f"EWM_S_{int(hl)}"])

        elif name == "PVO":
            pv_cols = [c for c in cols_all if str(c).startswith("PVO_") or c == "PVO"]
            add_if_present(pv_cols)
        elif name == "PVR":
            add_if_present("PVR")
        elif name == "BOP":
            add_if_present("BOP")
        elif name == "PXVOL":
            add_if_present(["DIR_STRENGTH","PXV_LR_VCHG","DIRxVOL"])

        elif name == "LOGRET":
            lags = kw.get("lags", [])
            lags = lags if isinstance(lags, (list, tuple)) else [lags]
            for k in lags:
                add_if_present(f"LOGRET_{int(k)}")
        elif name == "TIME_CYC":
            if bool(kw.get("daily", True)):
                add_if_present(["TOD_SIN","TOD_COS"])
            if bool(kw.get("weekly", True)):
                add_if_present(["DOW_SIN","DOW_COS"])

        elif name == "FOUND":
            add_if_present("funding_rate")
        elif name == "M15_DIR":
            add_if_present(["M15_DIR_01","M15_DIR_12","M15_DIR_23"])
        elif name == "M15_VOL":
            add_if_present(["M15_VOL_0","M15_VOL_1","M15_VOL_2","M15_VOL_3"])
        elif name == "FNG_IDX":
            add_if_present(["sent_fng","sent_fng_diff1","sent_fng_z7d"])
        elif name in {"OPEN","HIGH","LOW","CLOSE","VOLUME"}:
            # 僅在 plan 中被 enable 時才保留對應 OHLCV 欄位
            base = name.lower()
            if base in feat_df.columns:
                ohlcv_keep.add(base)
        else:
            # 未知名稱：忽略
            pass

    keep_set = set().union(ohlcv_keep).union(minute_cols).union(want)
    feat_cols = [c for c in cols_all if c in keep_set]
    return feat_cols

# ========== Fold Generator ==========
