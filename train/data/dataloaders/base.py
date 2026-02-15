# train/data/dataloaders/base.py
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from train.data.scalers import _get_scaler, ColumnSubsetScaler, pick_cols_to_scale


# ---------- I/O & Column selection ----------
def load_precomputed_features(
    *,
    path: Optional[str] = None,
    pre_feat_df: Optional[pd.DataFrame] = None,
    drop_time_cols: bool = True,
    drop_ohlcv: bool = True,
) -> pd.DataFrame:
    """
    載入離線預算特徵表並正規化索引為 UTC DatetimeIndex；若提供 pre_feat_df 則直接使用。
    - 支援 .csv / .parquet
    - 若含 'datetime' 或 'timestamp' 欄，會設為索引並移除該欄
    """
    if pre_feat_df is not None:
        df = pre_feat_df.copy()
    else:
        if not path:
            raise ValueError("需要提供 path 或 pre_feat_df")
        p = str(path)
        if p.endswith(".csv"):
            df = pd.read_csv(p)
        elif p.endswith(".parquet"):
            df = pd.read_parquet(p)
        else:
            raise ValueError("只支援 .csv 或 .parquet")

    if "datetime" in df.columns:
        idx = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        if drop_time_cols:
            df = df.drop(columns=["datetime"])
        df.index = idx
    elif "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
        unit = "ms" if (ts.dropna().iloc[0] if len(ts.dropna()) else 0) > 1_000_000_000_000 else "s"
        idx = pd.to_datetime(ts, unit=unit, utc=True)
        if drop_time_cols:
            df = df.drop(columns=["timestamp"])
        df.index = idx
    else:
        # 若原本就有 DatetimeIndex，也要統一成 UTC tz-aware
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("預算特徵表缺少 datetime/timestamp 且 index 不是 DatetimeIndex")
        df.index = ensure_utc_index(df.index)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # 再保險一次：全統一 UTC tz-aware
    df.index = ensure_utc_index(df.index)    
    # 最後只保留feature 把 "time" 跟原始 "ohlcv" 丟掉 (如果有要ohlcv在precomputed會有 _L1: shift(-1)版本的 )
    if drop_ohlcv:
        drop_cols = [col for col in ("open","high","low","close","volume") if col in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    if drop_time_cols:
        for col in ("datetime", "timestamp"):
            if col in df.columns:
                df = df.drop(columns=[col])
    return df

# ---------- Align & Fit-index ----------
def align_times(t0_index: pd.DatetimeIndex, idx_all: pd.DatetimeIndex, method: str) -> pd.DatetimeIndex:
    """
    將 t0_index 依 'exact' 或 'pad' 對齊到 idx_all（皆轉為 UTC tz-aware）。
    """
    method = str(method).lower()
    t0u = pd.DatetimeIndex(t0_index)
    if t0u.tz is None:
        t0u = t0u.tz_localize("UTC")
    else:
        t0u = t0u.tz_convert("UTC")
    if method == "exact":
        pos = idx_all.get_indexer(t0u)
        pos = pos[pos >= 0]
        return idx_all[pos]
    elif method == "pad":
        pos = idx_all.searchsorted(t0u, side="right") - 1
        pos = pos[pos >= 0]
        return idx_all[pos]
    else:
        raise ValueError("align_method must be 'exact' or 'pad'")


def fit_index_from_align(idx_all: pd.DatetimeIndex, align_idx: pd.DatetimeIndex, L: int) -> pd.DatetimeIndex:
    """
    由對齊後位置向量化產生 union 視窗 [p-L, p) 的索引，用於擬合 sklearn 縮放器。
    """
    if len(align_idx) == 0:
        return align_idx
    p_vec = idx_all.searchsorted(align_idx, side="right") - 1
    p_vec = p_vec[p_vec >= 0]
    if len(p_vec) == 0:
        return align_idx
    rng = np.arange(L, dtype=np.int32)
    fit_pos = (p_vec[:, None] - rng[None, :]).reshape(-1)
    fit_pos = fit_pos[(fit_pos >= 0) & (fit_pos < len(idx_all))]
    fit_pos = np.unique(fit_pos)
    return idx_all[fit_pos]

# ---------- 時區/索引 ----------
def ensure_utc_index(idx_like) -> pd.DatetimeIndex:
    """把任何 DatetimeIndex 或時間欄位轉為 UTC tz-aware 的 DatetimeIndex。"""
    idx = pd.DatetimeIndex(idx_like)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx


def flatten_micro_features(
    feat_df: pd.DataFrame,
    micro_df: Optional[pd.DataFrame],
    cv_start: pd.Timestamp,
    ts_end: pd.Timestamp,
    window_len: int = 15,
) -> pd.DataFrame:
    """
    將 1m micro 特徵展平成 m0~m(window_len-1) 並與 15m 特徵 join（m0 最舊、m(window_len-1) 最新）。
    - micro_df=None 或空表時直接回傳 feat_df
    - micro_df 會先裁剪到 [cv_start - window_len, ts_end]，再按 target_idx(<=ts_end) 做 reindex 展平
    - 不處理 NaN，呼叫端自行檢查
    """
    if micro_df is None or len(micro_df.index) == 0:
        return feat_df

    def _to_utc(ts_like):
        ts_obj = pd.Timestamp(ts_like)
        if ts_obj.tzinfo is None:
            return ts_obj.tz_localize("UTC")
        return ts_obj.tz_convert("UTC")

    cv_start = _to_utc(cv_start)
    ts_end = _to_utc(ts_end)

    micro_df = micro_df.sort_index()
    micro_df = micro_df.loc[(micro_df.index >= cv_start - pd.Timedelta(minutes=window_len)) & (micro_df.index <= ts_end)]
    if len(micro_df.index) == 0:
        return feat_df

    micro_cols = list(micro_df.columns)
    col_names: List[str] = []
    for i in range(window_len):
        for c in micro_cols:
            col_names.append(f"m{i}_{c}")

    target_idx = feat_df.index[feat_df.index <= ts_end]
    flat_rows = []
    for t in target_idx:
        t_utc = _to_utc(t)
        win_idx = pd.date_range(
            end=t_utc - pd.Timedelta(minutes=1),
            periods=window_len,
            freq="1min",
            tz="UTC",
        )
        sub = micro_df.reindex(win_idx)
        flat_rows.append(sub.to_numpy().reshape(-1))

    flat_df = pd.DataFrame(flat_rows, index=target_idx, columns=col_names, dtype=np.float32)
    return feat_df.join(flat_df, how="left")


# ---------- Scaling ----------
def apply_scaling(
    df: pd.DataFrame,
    feat_cols: List[str],
    scaler_kind: str,
    L: int,
    min_frac: float,
    fit_index: Optional[pd.DatetimeIndex] = None,
) -> Tuple[pd.DataFrame, Optional[ColumnSubsetScaler], List[str]]:
    """
    建立並套用縮放器：
      - time-safe scaler：直接 transform_full（不需 fit_index；內部自行 time-safe）
      - sklearn scaler：只在 fit_index（訓練覆蓋視窗）上 fit，然後 transform 全段
    回傳：(scaled_df, sklearn_scaler_or_None, cols_to_scale)
    """
    scaler = _get_scaler(scaler_kind, window=L, min_frac=min_frac)
    # 決定要縮放的欄位（跳過 sign-like / 一些 pattern）
    cols_to_scale = pick_cols_to_scale(df.loc[fit_index, feat_cols] if fit_index is not None else df[feat_cols], feat_cols)

    if hasattr(scaler, "is_timesafe") and scaler.is_timesafe:
        scaled = scaler.transform_full(df, cols_to_scale=cols_to_scale)
        return scaled, None, cols_to_scale

    if scaler is None:
        return df, None, cols_to_scale

    sklearn_scaler = ColumnSubsetScaler(scaler, all_cols=feat_cols, cols_to_scale=cols_to_scale)
    if fit_index is None or len(fit_index) == 0:
        raise ValueError("sklearn 縮放器需要 fit_index（通常由 train 視窗 union 計算）")
    sklearn_scaler.fit_df(df.loc[fit_index, feat_cols])
    arr = df.loc[:, feat_cols].values.astype(np.float32, copy=False)
    arr = sklearn_scaler.transform(arr)
    scaled = df.copy()
    scaled.loc[:, feat_cols] = arr
    return scaled, sklearn_scaler, cols_to_scale


# ---------- Misc helpers ----------
def label_counts_from_ds(ds) -> Dict[int, int]:
    """
    從 EventDataset/任意含 y 的對象計算二分類 0/1 計數；保證 0/1 key 存在。
    """
    import torch
    y = ds.y
    if isinstance(y, torch.Tensor):
        y = y.detach().to("cpu")
        uniq, cnt = torch.unique(y, return_counts=True)
        d = {int(u.item()): int(c.item()) for u, c in zip(uniq, cnt)}
    else:
        arr = np.asarray(y)
        u, c = np.unique(arr, return_counts=True)
        d = {int(uu): int(cc) for uu, cc in zip(u, c)}
    d.setdefault(0, 0); d.setdefault(1, 0)
    return d


def build_loaders(ds_tr, ds_va, ds_te, *, batch_size: int, device: str):
    """
    依據資料是否已在目標裝置，決定 pin_memory/num_workers，回傳三個 DataLoader。
    """
    pin = False  # EventDataset 通常預載到 device，不需 pin
    num_workers = 0
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
    return dl_tr, dl_va, dl_te

# ---------- 時間網格對齊（time-driven 專用） ----------
def reindex_to_full_grid(feat_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """以 data.freq 建立完整時間網格，並把 feat_df 對齊上去（僅 reindex，不補值）。"""
    if len(feat_df.index) == 0:
        return feat_df
    full_idx = pd.date_range(feat_df.index.min(), feat_df.index.max(), freq=str(freq), tz="UTC")
    return feat_df.reindex(full_idx)
