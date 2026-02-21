# train/data/dataloaders/base.py
from __future__ import annotations
from typing import Optional, List, Dict
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


# ---------- I/O & Column selection ----------
def load_precomputed_features(
    *,
    path: Optional[str] = None,
    pre_feat_df: Optional[pd.DataFrame] = None,
    drop_time_cols: bool = True,
    drop_ohlcv: bool = False,
    target_tz: str = "Asia/Taipei",
    assume_naive_tz: Optional[str] = None,
) -> pd.DataFrame:
    """
    載入離線預算特徵表並正規化索引為指定時區 DatetimeIndex；若提供 pre_feat_df 則直接使用。
    - 支援 .csv / .parquet
    - 若含 'datetime' / 'date' / 'timestamp' 欄，會設為索引並移除該欄
    """
    tz_target = resolve_timezone_name(target_tz)
    tz_assume = resolve_timezone_name(assume_naive_tz, default=tz_target) if assume_naive_tz else tz_target

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
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(tz_assume)
        idx = pd.DatetimeIndex(dt.dt.tz_convert(tz_target))
        if drop_time_cols:
            df = df.drop(columns=["datetime"])
        df.index = idx
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(tz_assume)
        idx = pd.DatetimeIndex(dt.dt.tz_convert(tz_target))
        if drop_time_cols:
            df = df.drop(columns=["date"])
        df.index = idx
    elif "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
        unit = "ms" if (ts.dropna().iloc[0] if len(ts.dropna()) else 0) > 1_000_000_000_000 else "s"
        idx = pd.DatetimeIndex(pd.to_datetime(ts, unit=unit, utc=True).tz_convert(tz_target))
        if drop_time_cols:
            df = df.drop(columns=["timestamp"])
        df.index = idx
    else:
        # 若原本就有 DatetimeIndex，也要統一成指定時區 tz-aware
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("預算特徵表缺少 datetime/timestamp 且 index 不是 DatetimeIndex")
        df.index = ensure_tz_index(df.index, target_tz=tz_target, assume_naive_tz=tz_assume)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # 再保險一次：全統一指定時區 tz-aware
    df.index = ensure_tz_index(df.index, target_tz=tz_target, assume_naive_tz=tz_assume)
    # 最後只保留feature 把 "time" 跟原始 "ohlcv" 丟掉 (如果有要ohlcv在precomputed會有 _L1: shift(-1)版本的 )
    if drop_ohlcv:
        drop_cols = [col for col in ("open","high","low","close","volume") if col in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    if drop_time_cols:
        for col in ("datetime", "date", "timestamp"):
            if col in df.columns:
                df = df.drop(columns=[col])
    return df


def load_tbm_events(*, path: str, parse_t1: bool = True) -> pd.DataFrame:
    """Load TBM events table from CSV or Parquet.

    Args:
        path: TBM file path. Supported suffixes are ``.csv`` and ``.parquet``.
        parse_t1: Whether to parse ``t1`` as datetime when the column exists.

    Returns:
        TBM dataframe with parsed datetime columns.
    """
    p = str(path)
    if p.endswith(".csv"):
        cols_head = pd.read_csv(p, nrows=0).columns
        parse_cols = ["t0"]
        if parse_t1 and "t1" in cols_head:
            parse_cols.append("t1")
        return pd.read_csv(p, parse_dates=parse_cols)

    if p.endswith(".parquet"):
        df = pd.read_parquet(p)
        if "t0" in df.columns:
            df["t0"] = pd.to_datetime(df["t0"], errors="coerce", utc=False)
        if parse_t1 and "t1" in df.columns:
            df["t1"] = pd.to_datetime(df["t1"], errors="coerce", utc=False)
        return df

    raise ValueError(f"Unsupported TBM file format: {path}. Only .csv/.parquet are supported.")


def load_ohlcv_for_label(
    *,
    path: str,
    target_tz: str = "Asia/Taipei",
    assume_naive_tz: Optional[str] = None,
) -> pd.DataFrame:
    """Load OHLCV data for labeling and validate required columns.

    Args:
        path: OHLCV source path.
        target_tz: Target timezone used in train pipeline.
        assume_naive_tz: Source timezone when datetime values are naive.
    Returns:
        OHLCV dataframe indexed by timezone-aware datetime index.
    """
    df = load_precomputed_features(
        path=path,
        drop_time_cols=True,
        drop_ohlcv=False,
        target_tz=target_tz,
        assume_naive_tz=assume_naive_tz,
    )
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"[ohlcv_fng_path] missing required columns: {missing}")
    out = df.loc[:, required].copy()
    return out.apply(pd.to_numeric, errors="coerce")

# ---------- Align ----------
def align_times(t0_index: pd.DatetimeIndex, idx_all: pd.DatetimeIndex, method: str) -> pd.DatetimeIndex:
    """
    將 t0_index 依 'exact' 或 'pad' 對齊到 idx_all（自動對齊 idx_all 時區）。
    """
    method = str(method).lower()
    idx_all = pd.DatetimeIndex(idx_all)
    idx_tz = idx_all.tz if idx_all.tz is not None else "UTC"
    t0u = pd.DatetimeIndex(t0_index)
    if t0u.tz is None:
        t0u = t0u.tz_localize(idx_tz)
    else:
        t0u = t0u.tz_convert(idx_tz)
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

# ---------- 時區/索引 ----------
def resolve_timezone_name(tz_value: Optional[str], default: str = "Asia/Taipei") -> str:
    """Resolve config timezone text into a valid timezone name.

    Args:
        tz_value: Raw timezone text from config.
        default: Fallback timezone.
    Returns:
        Valid timezone name.
    """
    if tz_value is None:
        return str(default)
    raw = str(tz_value).strip()
    if not raw:
        return str(default)

    m = re.search(r"\(([^)]+)\)", raw)
    if m:
        candidate = m.group(1).strip()
        if candidate:
            try:
                pd.Timestamp("2000-01-01", tz=candidate)
                return candidate
            except Exception:
                pass

    if re.search(r"utc\s*\+\s*08(?::?00)?", raw, flags=re.IGNORECASE):
        return "Asia/Taipei"

    try:
        pd.Timestamp("2000-01-01", tz=raw)
        return raw
    except Exception as exc:
        raise ValueError(f"Invalid timezone value: {tz_value}") from exc


def ensure_tz_index(
    idx_like,
    *,
    target_tz: str = "Asia/Taipei",
    assume_naive_tz: Optional[str] = None,
) -> pd.DatetimeIndex:
    """Convert datetime-like input to a timezone-aware DatetimeIndex.

    Args:
        idx_like: Datetime-like input.
        target_tz: Target timezone.
        assume_naive_tz: Source timezone when idx_like has no timezone.
    Returns:
        Timezone-aware DatetimeIndex in target_tz.
    """
    tz_target = resolve_timezone_name(target_tz)
    tz_assume = resolve_timezone_name(assume_naive_tz, default=tz_target) if assume_naive_tz else tz_target

    idx = pd.DatetimeIndex(idx_like)
    if idx.tz is None:
        idx = idx.tz_localize(tz_assume)
    else:
        idx = idx.tz_convert(tz_target)
    return pd.DatetimeIndex(idx)


def ensure_utc_index(idx_like) -> pd.DatetimeIndex:
    """Convert datetime-like input to UTC timezone-aware index.

    Args:
        idx_like: Datetime-like input.
    Returns:
        UTC timezone-aware DatetimeIndex.
    """
    return ensure_tz_index(idx_like, target_tz="UTC", assume_naive_tz="UTC")


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

    tz_target = pd.DatetimeIndex(feat_df.index).tz
    if tz_target is None and micro_df is not None:
        tz_target = pd.DatetimeIndex(micro_df.index).tz
    if tz_target is None:
        tz_target = "Asia/Taipei"

    def _to_tz(ts_like):
        """Convert timestamp-like object to feature timezone.

        Args:
            ts_like: Input timestamp-like value.
        Returns:
            Timezone-aware pandas Timestamp in tz_target.
        """
        ts_obj = pd.Timestamp(ts_like)
        if ts_obj.tzinfo is None:
            return ts_obj.tz_localize(tz_target)
        return ts_obj.tz_convert(tz_target)

    cv_start = _to_tz(cv_start)
    ts_end = _to_tz(ts_end)

    micro_df = micro_df.sort_index()
    micro_df = micro_df.loc[(micro_df.index >= cv_start - pd.Timedelta(minutes=window_len)) & (micro_df.index <= ts_end)]
    if len(micro_df.index) == 0:
        return feat_df
    micro_df = micro_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    micro_df = micro_df.ffill().fillna(0.0)

    micro_cols = list(micro_df.columns)
    col_names: List[str] = []
    for i in range(window_len):
        for c in micro_cols:
            col_names.append(f"m{i}_{c}")

    target_idx = feat_df.index[feat_df.index <= ts_end]
    flat_rows = []
    for t in target_idx:
        t_tz = _to_tz(t)
        win_idx = pd.date_range(
            end=t_tz - pd.Timedelta(minutes=1),
            periods=window_len,
            freq="1min",
            tz=tz_target,
        )
        sub = micro_df.reindex(win_idx)
        flat_rows.append(sub.to_numpy().reshape(-1))

    flat_df = pd.DataFrame(flat_rows, index=target_idx, columns=col_names, dtype=np.float32)
    return feat_df.join(flat_df, how="left")


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
    Build train/val/test loaders and auto-select pin_memory/num_workers.

    Args:
        ds_tr: Train dataset.
        ds_va: Validation dataset.
        ds_te: Test dataset.
        batch_size: Loader batch size.
        device: Runtime device string, e.g. ``cuda:0`` or ``cpu``.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    runtime_is_cuda = str(device).startswith("cuda")
    ds_device = getattr(getattr(ds_tr, "X", None), "device", None)
    ds_on_cpu = (ds_device is None) or (str(getattr(ds_device, "type", "cpu")) == "cpu")
    pin = bool(runtime_is_cuda and ds_on_cpu)
    num_workers = 10 if pin else 0
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin)
    return dl_tr, dl_va, dl_te

# ---------- 時間網格對齊（time-driven 專用） ----------
def reindex_to_full_grid(feat_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """以 data.freq 建立完整時間網格，並把 feat_df 對齊上去（僅 reindex，不補值）。"""
    if len(feat_df.index) == 0:
        return feat_df
    tz_target = pd.DatetimeIndex(feat_df.index).tz
    full_idx = pd.date_range(feat_df.index.min(), feat_df.index.max(), freq=str(freq), tz=tz_target)
    return feat_df.reindex(full_idx)
