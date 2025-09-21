# build_feature.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Optional, Dict

def _to_utc_index(idx, assume_tz: str = "UTC") -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(assume_tz)  # ← 讓呼叫端決定
    return idx.tz_convert("UTC")

# ------------------ FFD: Fixed-Width Fractional Differencing ------------------
def _ffd_weights(d: float, thres: float = 1e-5) -> np.ndarray:
    """
    生成固定視窗 FFD 權重 w，直到 |w_k| < thres 截斷。
    遞推：w[0]=1；w[k] = -w[k-1] * (d - k + 1) / k
    回傳 shape=(L,1)，順序為「舊→新」（對齊滾動窗）
    """
    if d < 0:
        raise ValueError("d must be >= 0 for fractional differencing.")
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    w = np.array(w[::-1], dtype="float64").reshape(-1, 1)
    return w  # 長度 L，將與長度 L 的窗口做內積

def fracDiff_FFD(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """
    對 1D 序列做固定視窗分數差分（不洩漏；僅用 <= t）。
    回傳與輸入等長的 Series（前段為 NaN）。
    常用：對 log 價先做 FFD，得到近似平穩且保留記憶的序列。
    """
    s = pd.Series(series, copy=True).astype("float64")
    s = s.replace([np.inf, -np.inf], np.nan).ffill()  # 保守處理
    w = _ffd_weights(d, thres)                        # shape=(L,1)
    L = len(w)                                        # 需要 L 個點
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    vals = s.values.astype("float64")
    # 對每個 t，使用 vals[t-L+1 : t+1] 與 w 內積
    for t in range(L - 1, len(vals)):
        window = vals[t - L + 1: t + 1].reshape(-1, 1)
        out.iloc[t] = float(np.dot(w.T, window))

    return out

# ------------------ read TBM precomputed label ------------------
def assign_tbm_labels_to_df(
    df_15m: pd.DataFrame,
    tbm_csv_path: str,
    *,
    keep_sides: Literal["both", "long", "short"] = "both",
    drop_nan_labels: bool = True,
    align_method: Literal["exact", "pad"] = "exact",
    col_name: str = "y_cls",
    dedup_strategy: Literal["prefer_fail", "prefer_win", "earliest_t1", "latest_t1"] = "prefer_fail",
    assume_tz: str = "UTC",   # 若 TBM 檔是 naive 時間，當作哪個時區來 localize
) -> pd.DataFrame:
    tbm = pd.read_csv(tbm_csv_path, parse_dates=["t0", "t1"])
    req = {"t0", "t1", "label", "side", "entry_price"}
    missing = req - set(tbm.columns)
    if missing:
        raise ValueError(f"TBM CSV 缺少欄位: {sorted(missing)}")

    # 轉成同一時區（UTC）以便對齊
    tbm["t0"] = _to_utc_index(tbm["t0"], assume_tz=assume_tz)
    tbm["t1"] = _to_utc_index(tbm["t1"], assume_tz=assume_tz)

    # side → ±1
    if tbm["side"].dtype == object:
        tbm["side"] = tbm["side"].map({"Long": 1, "Short": -1}).astype("Int8")
    else:
        tbm["side"] = tbm["side"].astype("Int8")

    # 篩 side
    if keep_sides == "long":
        tbm = tbm[tbm["side"] == 1]
    elif keep_sides == "short":
        tbm = tbm[tbm["side"] == -1]

    # 丟 NaN 標籤
    if drop_nan_labels:
        tbm = tbm[tbm["label"].notna()]

    # 先去掉完全重複的 (t0, side, label) 列
    before = len(tbm)
    tbm = tbm.drop_duplicates(subset=["t0", "side", "label"], keep="last")
    # 若同一 (t0, side) 仍有多筆（代表 label 或 t1 不同），依策略擇一
    if tbm.duplicated(subset=["t0", "side"]).any():
        if dedup_strategy == "prefer_fail":
            # 失敗優先（保守）；若同 label 再比 t1（早者先）
            tbm = (tbm
                   .sort_values(["t0","side","label","t1"], ascending=[True, True, True, True])
                   .drop_duplicates(["t0","side"], keep="first"))
        elif dedup_strategy == "prefer_win":
            tbm = (tbm
                   .sort_values(["t0","side","label","t1"], ascending=[True, True, True, True])
                   .drop_duplicates(["t0","side"], keep="last"))
        elif dedup_strategy == "earliest_t1":
            tbm = (tbm
                   .sort_values(["t0","side","t1"])
                   .drop_duplicates(["t0","side"], keep="first"))
        elif dedup_strategy == "latest_t1":
            tbm = (tbm
                   .sort_values(["t0","side","t1"])
                   .drop_duplicates(["t0","side"], keep="last"))
        else:
            raise ValueError(f"Unknown dedup_strategy: {dedup_strategy}")

    # 若 keep_sides=="both"，此時可能同一 t0 仍有兩筆（long/short 各一）。
    # 你的目標是把 label 貼到單一 15m 列 → 無法同時容納兩筆，請選邊。
    if keep_sides == "both" and tbm.duplicated(subset=["t0"]).any():
        dups = tbm.loc[tbm.duplicated(subset=["t0"], keep=False), ["t0", "side", "label"]]
        raise ValueError(
            "同一 t0 同時存在 long/short。請設定 keep_sides='long' 或 'short'。\n"
            f"重複示例：\n{dups.head(10)}"
        )

    # 建立 (t0 -> label) 對映
    lab_map = tbm.set_index("t0")["label"].astype(int)

    # 對齊 15m 索引
    df_idx = pd.DatetimeIndex(df_15m.index).sort_values()
    t0_idx = pd.DatetimeIndex(lab_map.index)
    if align_method == "exact":
        can_align = t0_idx.isin(df_idx)
        lab_map = lab_map[can_align]
        assign_index = pd.DatetimeIndex(lab_map.index)
    elif align_method == "pad":
        pos = df_idx.searchsorted(t0_idx, side="right") - 1
        # 過濾掉「在資料最前面之前」的 t0
        valid = pos >= 0
        pos = np.clip(pos, 0, len(df_idx)-1)
        assign_index = df_idx[pos][valid]
        lab_map = lab_map[valid]
    else:
        raise ValueError("align_method must be 'exact' or 'pad'.")

    # 寫入 df[col_name]
    out = df_15m.copy()
    out[col_name] = np.nan
    out.loc[assign_index, col_name] = lab_map.values
    out[col_name] = out[col_name].astype("float")  # 0/1 + NaN
    return out

def create_labels_adaptive(df: pd.DataFrame,
                           mode: str = "vol",          # "vol" | "event_TBM"
                           flat_band_bps: float = 5.0, # for mode="bps"
                           k_vol: float = 0.5,         # for mode="vol"
                           vol_window: int = 96,       # 以 1H 資料為例，過去 4 天
                           q_flat: float = 0.3,        # for mode="quantile": 中間 40% = 持平
                           q_min_obs: int = 500,       # for mode="quantile": 展開期最少樣本數
                           roundtrip_cost_bps: float = 10.0, # for mode="cost"
                           triple_k: float = 0.5,      # for mode="triple"
                           triple_window: int = 96,

                           ret_shift: int | None = None,# 如果是1H就 = 1；如果是15 min就 = 4 (4根15m=1H)
                           ret_type: str | None = "logret",   # fractionally
                           ffd_d: float = 0.3,
                           ffd_thres: float = 1e-5,

                            # --- for "event_TBM" ---
                            tbm_csv_path: Optional[str] = None,
                            keep_sides: str = "both",
                            drop_nan_labels: bool = True,
                            align_method: str = "exact"
                           ): 
    """
    回傳 df，含：
      - y_reg: 未來 ret_shift 根的目標
               當 ret_type="logret" → 一般對數報酬
               當 ret_type="fractionally" → 分數差分空間的未來「位移」
      - y_cls: 三分類（0=Down, 1=Flat, 2=Up）
    """

    df = df.copy()

    if mode == "event_tbm":
        if not tbm_csv_path:
            raise ValueError("mode='event_TBM' 需要 tbm_csv_path")
        # 直接把 TBM 的 0/1 貼到 df['y_cls']（二分類）
        df = assign_tbm_labels_to_df(
            df_15m=df,
            tbm_csv_path=tbm_csv_path,
            keep_sides=keep_sides,
            drop_nan_labels=drop_nan_labels,
            align_method=align_method,
            col_name="y_cls"
        )
        return df

    close = df["close"].astype(float)
    fwd_k = int(ret_shift)

    # ----------- y_reg：未來 fwd_k 根目標 ----------- 
    logret_fwd = np.log(close.shift(-fwd_k)) - np.log(close)  # 供分類門檻使用（傳統空間）
    if ret_type == "logret":
        df["y_reg"] = logret_fwd

    elif ret_type == "fractionally":
        # 1) 對 log 價做固定視窗 FFD（只用 ≤t 資料，不洩漏）
        logp = np.log(close)
        logp_fd = fracDiff_FFD(logp, d=ffd_d, thres=ffd_thres)
        # 2) 在 FFD 空間定義「未來 fwd_k 根位移」作為回歸目標
        y_fd = logp_fd.shift(-fwd_k) - logp_fd
        df["y_reg"] = y_fd

    else:
        raise ValueError(f"Unknown ret_type: {ret_type}")


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

def create_label(df_base: pd.DataFrame, cfg: Dict, return_what: str = "auto") -> pd.Series:
    """
    Create a single label series based on cfg and return_what.

    - For mode='event_tbm': aligns TBM CSV onto the 15m grid and returns 'y_cls'.
    - For time-driven modes: computes either 'y_reg' or 'y_cls' according to task type
      or explicitly via return_what in {'reg','cls'}.

    Returns
    -------
    pd.Series
        Named 'y_reg' or 'y_cls'. Index matches df_base after processing.
    """
    # Use the same logic as build_features_and_label but only return one series
    Lcfg = cfg["label"]
    mode = str(Lcfg.get("mode", "vol")).lower()

    if mode == "event_tbm":
        df_lab = assign_tbm_labels_to_df(
            df_15m=df_base,
            tbm_csv_path=Lcfg["tbm_csv_path"],
            keep_sides=Lcfg.get("keep_sides", "both"),
            drop_nan_labels=Lcfg.get("drop_nan_labels", True),
            align_method=Lcfg.get("align_method", "exact"),
        )
        s = df_lab["y_cls"].copy()
        s.name = "y_cls"
        return s

    # time-driven
    task_type = str(cfg["task"]["type"]).lower()
    want = return_what.strip().lower() if isinstance(return_what, str) else "auto"
    want_cls = (want == "cls") or (want == "auto" and task_type == "classification")

    # Reuse implementation from build_features_and_label to ensure parity
    tmp = create_labels_adaptive(
        df_base.copy(),
        mode=mode,
        flat_band_bps=float(Lcfg.get("flat_band_bps", 5.0)),
        roundtrip_cost_bps=float(Lcfg.get("roundtrip_cost_bps", 10.0)),
        k_vol=float(Lcfg.get("k_vol", 0.5)),
        vol_window=int(Lcfg.get("vol_window", 96)),
        q_flat=float(Lcfg.get("q_flat", 0.3)),
        q_min_obs=int(Lcfg.get("q_min_obs", 500)),
        triple_k=float(Lcfg.get("triple_k", 0.5)),
        triple_window=int(Lcfg.get("triple_window", 96)),
        ret_shift=int(Lcfg["ret_shift"]),
        ret_type=str(Lcfg.get("ret_type", "logret")).lower(),
        ffd_d=float(Lcfg.get("fracdiff", {}).get("d", 0.3)),
        ffd_thres=float(Lcfg.get("fracdiff", {}).get("thres", 1e-5)),
    )
    if want_cls:
        s = tmp["y_cls"].copy()
        s.name = "y_cls"
        return s
    else:
        s = tmp["y_reg"].copy()
        s.name = "y_reg"
        return s
