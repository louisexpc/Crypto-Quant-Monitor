# build_feature.py
import numpy as np
import pandas as pd
from pathlib import Path


def _to_utc_index(idx, assume_tz: str = "UTC") -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(assume_tz)  # ← 讓呼叫端決定
    return idx.tz_convert("UTC")

def create_labels_adaptive(df: pd.DataFrame,
                           horizon: int = 1,
                           mode: str = "vol",          # "bps" | "vol" | "quantile" | "cost" | "triple"
                           flat_band_bps: float = 5.0, # for mode="bps"
                           k_vol: float = 0.5,         # for mode="vol"
                           vol_window: int = 96,       # 以 1H 資料為例，過去 4 天
                           q_flat: float = 0.3,        # for mode="quantile": 中間 40% = 持平
                           q_min_obs: int = 500,       # for mode="quantile": 展開期最少樣本數
                           roundtrip_cost_bps: float = 10.0, # for mode="cost"
                           triple_k: float = 0.5,      # for mode="triple"
                           triple_window: int = 96,
                           ret_shift: int | None = None# 如果是1H就 = 1；如果是15 min就 = 4 (4根15m=1H)
                           ): 
    """
    回傳 df，含：
      - y_reg: 未來 ret_shift 根（若 ret_shift=None 則用 horizon 根）的 log return
      - y_cls: 三分類（0=Down, 1=Flat, 2=Up）
    說明：
      - ret_shift 控制 y_reg 使用的「前瞻步數」；例如 15m 資料預測 1h → ret_shift=4；1H 資料 → ret_shift=1
      - 各種門檻（例如 vol 模式）會依 ret_shift 做尺度調整（√ret_shift）
    """

    df = df.copy()
    close = df["close"].astype(float)
    fwd_k = int(ret_shift if ret_shift is not None else horizon)

    # ----------- y_reg：未來 fwd_k 根報酬 -----------
    logret_fwd = np.log(close.shift(-fwd_k)) - np.log(close)
    df["y_reg"] = logret_fwd

    # ----------- 計算 threshold_t（定義 Flat 的幅度）-----------
    if mode == "bps":
        thr = flat_band_bps / 10000.0
        thr_t = pd.Series(thr, index=df.index).shift(1)  # 防洩漏

    elif mode == "cost":
        thr = roundtrip_cost_bps / 10000.0
        thr_t = pd.Series(thr, index=df.index).shift(1)

    elif mode == "vol":
        lr_past = np.log(close).diff()
        sigma = lr_past.rolling(vol_window, min_periods=max(2, vol_window//2)).std()
        # 對應 fwd_k 根的尺度（布朗縮放假設）
        thr_t = (k_vol * sigma * np.sqrt(fwd_k)).reindex(df.index).shift(1)

    elif mode == "quantile":
        # 用「過去 fwd_k 根」的報酬分佈決定閾值（不洩漏）
        lr_past = (np.log(close).shift(-fwd_k) - np.log(close)).shift(+fwd_k)
        q_low  = lr_past.expanding(min_periods=q_min_obs).quantile(q_flat)
        q_high = lr_past.expanding(min_periods=q_min_obs).quantile(1 - q_flat)
        thr_t = pd.concat([q_low.abs(), q_high.abs()], axis=1).max(axis=1).shift(1)

    elif mode == "triple":
        # 以 fwd_k 作為觀察窗長度
        lr_past = np.log(close).diff()
        sigma = lr_past.rolling(triple_window, min_periods=max(2, triple_window//2)).std()
        up_bar = triple_k * sigma
        dn_bar = -triple_k * sigma

        y_cls = np.full(len(df), -1, dtype=np.int8)
        for t in range(len(df) - fwd_k):
            p0 = np.log(close.iloc[t])
            rel = np.log(close.iloc[t+1:t+1+fwd_k].values) - p0
            ub = up_bar.iloc[t]; db = dn_bar.iloc[t]
            if np.isnan(ub) or np.isnan(db):
                continue
            hit_up = (rel >= ub).any()
            hit_dn = (rel <= db).any()
            if hit_up and not hit_dn: y_cls[t] = 2
            elif hit_dn and not hit_up: y_cls[t] = 0
            else: y_cls[t] = 1
        df["y_cls"] = y_cls
        return df

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ----------- 依 threshold_t 標註三分類（0/1/2）-----------
    y_cls = np.full(len(df), -1, dtype=int)
    cond_up   = logret_fwd >  thr_t
    cond_down = logret_fwd < -thr_t
    cond_flat = (~cond_up) & (~cond_down)

    y_cls[cond_up.fillna(False).values]   = 2
    y_cls[cond_down.fillna(False).values] = 0
    y_cls[cond_flat.fillna(False).values] = 1

    df["y_cls"] = y_cls
    return df



def build_features_and_label(
    df_base: pd.DataFrame,
    feat_parquet_path: str | None = None,
    feat_df: pd.DataFrame | None = None,
    *,
    cfg
    ):
    """
    對齊你的原版行為（UTC/網格/shift），但：
    - regression：y 為連續報酬（未來 horizon 的 log 或 simple）
    - classification：y 為3值 (ret > cls_threshold)
    """

    # === 0) 時間設定 ===
    start_date = pd.Timestamp(cfg["cv"]["start_date"]).tz_localize("UTC")
    end_date   = pd.Timestamp(cfg["cv"]["end_date"]).tz_localize("UTC")
    freq = cfg["data"]["freq"]

    # === 1) 讀特徵 ===
    if feat_df is None:
        if not feat_parquet_path or not Path(feat_parquet_path).exists():
            raise FileNotFoundError(f"特徵檔不存在：{feat_parquet_path}")
        X = pd.read_parquet(feat_parquet_path)
    else:
        X = feat_df.copy()

    # 安全：移除任何 label-like 欄位
    label_like = {"label", "target", "y", "y_cls", "y_reg"}
    bad = [c for c in X.columns if c.lower() in label_like]
    if bad:
        print(f"[SAFE] Dropping label-like columns from features: {bad}")
        X = X.drop(columns=bad)

    # === 2) 對齊索引/排序 ===
    dfb = df_base.copy()
    X.index   = _to_utc_index(X.index)
    dfb.index = _to_utc_index(dfb.index)
    X   = X.sort_index()
    dfb = dfb.sort_index()
    X   = X[~X.index.duplicated(keep="last")]
    dfb = dfb[~dfb.index.duplicated(keep="last")]

    # 3) 強制完整 1H 網格
    full_idx = pd.date_range(dfb.index.min(), dfb.index.max(), freq=str(freq), tz="UTC")
    dfb = dfb.reindex(full_idx)
    X   = X.reindex(full_idx)
        
    # === 4) 產生 y（用 create_labels_adaptive 一次搞定；支援 ret_shift） ===
    Lcfg = cfg["label"]
    horizon = int(Lcfg["horizon"])
    ret_shift = int(Lcfg["ret_shift"])
    task_type = str(cfg["task"]["type"]).lower()
    mode = str(Lcfg["mode"]).lower()

    # 安全檢查
    if not {"open","high","low","close","volume"}.issubset(dfb.columns):
        raise KeyError("df_base 缺少 OHLCV 欄位")

    # 呼叫一次產生 y_reg(=logret over ret_shift) 與 y_cls
    df_lbl = create_labels_adaptive(
        dfb,
        horizon=horizon,
        mode=mode,
        flat_band_bps=float(Lcfg.get("flat_band_bps", 5.0)),
        roundtrip_cost_bps=float(Lcfg.get("roundtrip_cost_bps", 10.0)),
        k_vol=float(Lcfg.get("k_vol", 0.5)),
        vol_window=int(Lcfg.get("vol_window", 96)),
        q_flat=float(Lcfg.get("q_flat", 0.3)),
        q_min_obs=int(Lcfg.get("q_min_obs", 500)),
        triple_k=float(Lcfg.get("triple_k", 0.5)),
        triple_window=int(Lcfg.get("triple_window", 96)),
        ret_shift=ret_shift,   # ★ 關鍵
    )

    # === 5) 依任務挑 y（避免重算）===
    if task_type == "classification":
        y = df_lbl["y_cls"].rename("label")
    else:
        # regression：若要 simple return，就由 logret 轉換；否則直接用 logret
        y = df_lbl["y_reg"].astype("float32").rename("target")

    # === 6) 去掉未來 close 為 nan 或特徵不完整的時點 ===
    valid_now = X.notna().all(axis=1)
    valid_lbl = y.notna()
    keep = valid_now & valid_lbl
    X, y = X[keep], y[keep]

    # === 7) 篩選時間區間（最重要）===
    mask_range = (X.index >= start_date) & (X.index <= end_date)
    X = X.loc[mask_range]
    y = y.loc[mask_range]

    # === 8) 再次清理數值（NaN / inf） ===
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    X = X.loc[y.index]

    # === 9) 最後過濾不合法的 label（分類限定）
    if task_type == "classification":
        # 轉回整數類別並過濾非法值
        y = y.astype("int")
        num_classes = int(cfg["model"]["num_classes"])
        y = y[(y >= 0) & (y < num_classes)]
        X = X.loc[y.index]
    
    X, y = X.align(y, join="inner", axis=0)
    # --- 回傳前再做一次保險檢查 ---
    bad2 = [c for c in X.columns if c.lower() in label_like]
    assert not bad2, f"X 仍包含標籤欄位：{bad2}"
    return X, y


