import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _to_utc_index(idx: pd.Index | pd.Series, assume_tz: str = "UTC") -> pd.DatetimeIndex:
    """Normalize any time-like index/series to UTC tz-aware DatetimeIndex.

    - If input already tz-aware: convert to UTC.
    - If naive: localize to `assume_tz` then convert to UTC.
    """
    di = pd.DatetimeIndex(idx)
    if di.tz is None:
        di = di.tz_localize(assume_tz)
    return di.tz_convert("UTC")


def _drop_label_like_cols(df: pd.DataFrame) -> pd.DataFrame:
    bad = {"label", "target", "y", "y_cls", "y_reg"}
    cols = [c for c in df.columns if str(c).lower() not in bad]
    return df.loc[:, cols]


@dataclass
class EventRow:
    t0_raw: pd.Timestamp        # 原 CSV 的 t0（可能含原時區）
    t0_utc: pd.Timestamp        # 轉為 UTC 的 t0（用於對齊）
    t0_align: pd.Timestamp      # 對齊到 15m 網格的時間點（不含 t0 當根）
    t1_utc: Optional[pd.Timestamp]
    side: int                   # +1 / -1
    label: int                  # 0 / 1


class EventDataset(Dataset):
    """
    Build per-event sequences from a 15m feature grid:

    - For each event t0, take the previous `seq_len` bars strictly before the aligned t0 bar.
    - Labels come from a TBM label CSV (columns: t0, t1, label, side, entry_price ...).
    - Feature dataframe must be indexed on the 15m UTC grid (tz-aware), sorted, deduped.

    Notes
    -----
    - align_method='exact': require t0 to be exactly on the 15m grid; otherwise event is dropped.
    - align_method='pad': use the last bar strictly before t0 (i.e., floor to grid).
    - If a window contains any NaNs (or insufficient history), the event is dropped.
    """

    def __init__(
        self,
        feat_df: pd.DataFrame,
        tbm_csv_path: str,
        *,
        seq_len: int = 144,
        feature_cols: Optional[List[str]] = None,
        keep_sides: Literal["both", "long", "short"] = "both",
        align_method: Literal["exact", "pad"] = "pad",
        assume_tz: str = "UTC",
        drop_incomplete: bool = True,
        device: Optional[str] = None,
        allowed_align_index: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        super().__init__()

        if not isinstance(feat_df.index, pd.DatetimeIndex):
            raise TypeError("feat_df.index must be a DatetimeIndex")

        # 1) Normalize features: UTC grid, sorted, drop dup, drop label-like
        X = feat_df.copy()
        # If tz-naive, localize to assume_tz and convert to UTC
        if X.index.tz is None:
            X.index = _to_utc_index(X.index, assume_tz=assume_tz)
        else:
            X.index = X.index.tz_convert("UTC")
        X = X.sort_index()
        X = X[~X.index.duplicated(keep="last")]
        X = _drop_label_like_cols(X)

        # Select feature columns
        if feature_cols is None:
            # by default: all numeric columns
            num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            feature_cols = num_cols
        else:
            feature_cols = [c for c in feature_cols if c in X.columns]
        if not feature_cols:
            raise ValueError("No valid feature columns selected for EventDataset")

        # 2) Read TBM events
        if not os.path.exists(tbm_csv_path):
            raise FileNotFoundError(f"TBM label CSV not found: {tbm_csv_path}")
        tbm = pd.read_csv(tbm_csv_path, parse_dates=["t0", "t1"])
        req = {"t0", "label", "side"}
        missing = req - set(tbm.columns)
        if missing:
            raise ValueError(f"TBM CSV missing columns: {sorted(missing)}")

        # to UTC
        t0_utc = _to_utc_index(tbm["t0"], assume_tz=assume_tz)
        t1_utc = _to_utc_index(tbm["t1"], assume_tz=assume_tz)
        tbm = tbm.assign(t0_utc=t0_utc, t1_utc=t1_utc)

        # side normalization
        if tbm["side"].dtype == object:
            tbm["side"] = tbm["side"].map({"Long": 1, "Short": -1}).astype("Int8")
        else:
            tbm["side"] = tbm["side"].astype("Int8")
        tbm["label"] = tbm["label"].astype(float)
        tbm = tbm[tbm["label"].notna()].copy()
        tbm["label"] = tbm["label"].astype(int)

        if keep_sides == "long":
            tbm = tbm[tbm["side"] == 1]
        elif keep_sides == "short":
            tbm = tbm[tbm["side"] == -1]

        # 3) Align t0 to feature grid and build sequences
        idx = pd.DatetimeIndex(X.index)
        L = int(seq_len)
        feats = X.loc[:, feature_cols].astype(np.float32)

        rows: List[EventRow] = []
        seqs: List[np.ndarray] = []
        labels: List[int] = []

        # drop 統計
        total_events = 0
        drop_short = 0
        drop_nan = 0
        drop_outside = 0

        # vectorized searchsorted for t0 alignment
        t0u = pd.DatetimeIndex(tbm["t0_utc"])  # UTC
        if align_method == "exact":
            # keep only events exactly on grid
            mask_on_grid = t0u.isin(idx)
            t0u = t0u[mask_on_grid]
            tbm = tbm.loc[mask_on_grid]
            pos = idx.get_indexer(t0u)
        elif align_method == "pad":
            # floor to previous bar: right-1
            pos = idx.searchsorted(t0u, side="right") - 1
            # drop those before the first bar
            valid = pos >= 0
            pos = pos[valid]
            tbm = tbm.loc[valid]
        else:
            raise ValueError("align_method must be 'exact' or 'pad'")

        # Precompute allowed align set for fast filtering
        allowed_set = None
        if allowed_align_index is not None:
            allowed_set = set(pd.DatetimeIndex(allowed_align_index))

        # Build sequences
        for i, p in enumerate(pos):
            total_events += 1
            if p < L:
                if drop_incomplete:
                    drop_short += 1
                    continue
                else:
                    # left-pad with NaNs if allowed (we default to drop)
                    seq_window = feats.iloc[0:p].values
                    if len(seq_window) == 0:
                        continue
                    pad = np.full((L - len(seq_window), seq_window.shape[1]), np.nan, dtype=np.float32)
                    arr = np.concatenate([pad, seq_window], axis=0)
            else:
                arr = feats.iloc[p - L:p].values  # strictly before align time

            if np.isnan(arr).any():
                if drop_incomplete:
                    drop_nan += 1
                    continue
                # else leave NaNs (not recommended)

            align_time = idx[p]
            if allowed_set is not None and align_time not in allowed_set:
                drop_outside += 1
                continue

            ev = EventRow(
                t0_raw=pd.Timestamp(tbm.iloc[i]["t0"]),
                t0_utc=pd.Timestamp(tbm.iloc[i]["t0_utc"]),
                t0_align=align_time,
                t1_utc=pd.Timestamp(tbm.iloc[i]["t1_utc"]) if pd.notna(tbm.iloc[i]["t1_utc"]) else None,
                side=int(tbm.iloc[i]["side"]),
                label=int(tbm.iloc[i]["label"]),
            )

            rows.append(ev)
            seqs.append(arr.astype(np.float32, copy=False))
            labels.append(ev.label)

        if not seqs:
            raise ValueError("No valid events after alignment, history, and NaN checks.")

        # 印出統計
        kept = len(seqs)
        print(f"[EventDataset] total={total_events} | kept={kept} | drop_short={drop_short} | drop_nan={drop_nan} | drop_outside={drop_outside}")

        X_np = np.stack(seqs, axis=0)  # [N, L, F]
        y_np = np.array(labels, dtype=np.int64)  # [N]

        # default device: cuda if available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.X = torch.tensor(X_np, dtype=torch.float32, device=device)
        self.y = torch.tensor(y_np, dtype=torch.long, device=device)
        self.events: List[EventRow] = rows
        self.feature_cols = feature_cols
        self.seq_len = L
        self.device = device

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]

    # Convenience accessors
    def event_times(self) -> List[pd.Timestamp]:
        return [r.t0_align for r in self.events]

    def event_meta(self) -> List[Dict[str, object]]:
        out = []
        for r in self.events:
            out.append({
                "t0_raw": r.t0_raw,
                "t0_utc": r.t0_utc,
                "t0_align": r.t0_align,
                "t1_utc": r.t1_utc,
                "side": r.side,
                "label": r.label,
            })
        return out


def load_features_csv(
    path: str,
    *,
    time_col: str = "datetime",
    assume_tz: str = "UTC",
) -> pd.DataFrame:
    """Load a CSV features file and return a UTC-indexed DataFrame.

    - Expects a time column (default 'datetime').
    - If time values are tz-naive, localize to `assume_tz` then convert to UTC.
    - Keeps all columns as-is (caller chooses feature subset later).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise KeyError(f"CSV missing time_col='{time_col}'")
    idx = _to_utc_index(pd.to_datetime(df[time_col], errors="coerce"), assume_tz=assume_tz)
    df = df.drop(columns=[time_col])
    df.index = idx
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ============================
# Builder via FeatureComputer
# ============================
def build_event_dataset_via_feature_computer(
    cfg: dict,
    tbm_csv_path: str,
    *,
    seq_len: int = 144,
    keep_sides: Literal["both", "long", "short"] = "both",
    align_method: Literal["exact", "pad"] = "pad",
    device: Optional[str] = None,
) -> EventDataset:
    """Compute 15m features via FeatureComputer, then assemble EventDataset.

    Uses cfg sections:
    - data.path           : OHLCV CSV/Parquet path
    - data.index_col      : name of time column in the OHLCV file
    - data.freq           : expected frequency (e.g., "15min")
    - features.cache_dir  : cache dir for FeatureComputer
    - features.plan       : feature plan definition
    - env: WORKER_TAG     : optional subdir for cache isolation
    """
    from .indicators import IndicatorLibrary, FeatureComputer
    import os

    raw_path = cfg["data"]["path"]
    index_col = cfg["data"]["index_col"]
    freq = cfg["data"]["freq"]

    # Load raw OHLCV and normalize
    if str(raw_path).endswith(".csv"):
        raw_df = pd.read_csv(raw_path)
    elif str(raw_path).endswith(".parquet"):
        raw_df = pd.read_parquet(raw_path)
    else:
        raise ValueError("Only .csv or .parquet is supported for data.path")

    lib = IndicatorLibrary(raw_df, freq_check=freq, prefer_time_col=index_col)

    # Setup feature computer with cache and worker tag
    cache_dir = cfg["features"]["cache_dir"]
    worker_tag = os.environ.get("WORKER_TAG", "").strip()
    if worker_tag:
        cache_dir = os.path.join(cache_dir, worker_tag)
    fc = FeatureComputer(lib, cache_dir=cache_dir)

    # Compute features according to plan (already shift(1) inside)
    plan = cfg["features"]["plan"]
    feat_df = fc.compute(plan, cfg)

    # Assemble dataset
    ds = EventDataset(
        feat_df=feat_df,
        tbm_csv_path=tbm_csv_path,
        seq_len=seq_len,
        feature_cols=feat_df.columns.tolist(),
        keep_sides=keep_sides,
        align_method=align_method,
        device=device,
    )
    return ds
