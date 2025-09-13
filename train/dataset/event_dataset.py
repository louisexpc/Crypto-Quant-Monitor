import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _to_utc_index(idx: pd.Index | pd.Series, assume_tz: str = "UTC") -> pd.DatetimeIndex:
    """
    1. 說明:
        將任何時間型態的 Index/Series 正規化成「UTC」時區的 tz-aware DatetimeIndex。
        - 若來源已帶時區：直接轉換為 UTC。
        - 若來源為 naive（無時區）：先以 assume_tz 本地化，再轉為 UTC。
    2. inputs:
        - idx: pd.Index | pd.Series
        - assume_tz: str = "UTC"
    3. return:
        - pd.DatetimeIndex (tz='UTC')
    """
    di = pd.DatetimeIndex(idx)
    if di.tz is None:
        di = di.tz_localize(assume_tz)
    return di.tz_convert("UTC")


def _drop_label_like_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. 說明:
        過濾掉疑似標籤欄位，避免把 label/target 混入特徵矩陣。
    2. inputs:
        - df: pd.DataFrame
    3. return:
        - pd.DataFrame
    """
    bad = {"label", "target", "y", "y_cls", "y_reg"}
    cols = [c for c in df.columns if str(c).lower() not in bad]
    return df.loc[:, cols]


@dataclass
class EventRow:
    """
    1. 說明:
        TBM 事件的基本欄位與對齊資訊（供 meta 使用）。
    2. inputs:
        - t0_raw, t0_utc, t0_align, t1_utc, side, label
    3. return:
        - dataclass 物件
    """
    t0_raw: pd.Timestamp        # 原 CSV 的 t0（可能含原時區）
    t0_utc: pd.Timestamp        # 轉為 UTC 的 t0（用於對齊）
    t0_align: pd.Timestamp      # 對齊到 15m 網格的時間點（不含 t0 當根）
    t1_utc: Optional[pd.Timestamp]
    side: int                   # +1 / -1
    label: int                  # 0 / 1


class EventDataset(Dataset):
    """
    1. 說明:
        從 15m 固定網格的特徵表中，依 TBM 事件組出每事件的歷史序列視窗（僅取 t0_align 之前的 seq_len 根）。
        標籤 y 來自 TBM 的 label（0/1），方向 side 留在 meta。
        本實作會在 __init__ 即將 [N,L,F] 與 [N] 預載到指定 device（含 GPU）。

    2. inputs:
        - feat_df: pd.DataFrame（index=DatetimeIndex）
        - tbm_csv_path: str（TBM CSV 路徑，至少含 t0/label/side）
        - seq_len: int = 144
        - feature_cols: Optional[List[str]] = None（None 則自動取數值欄）
        - keep_sides: {"both","long","short"} = "both"
        - align_method: {"exact","pad"} = "pad"（exact: t0 必在網格；pad: 對齊到前一根）
        - assume_tz: str = "UTC"（naive 時區假定）
        - drop_incomplete: bool = True（有缺/不足即丟棄；False 則左側 NaN 補齊並 ffill）
        - device: Optional[str] = None（None 自動選 cuda/ cpu）
        - allowed_align_index: Optional[pd.DatetimeIndex] = None（若提供，只保留其內對齊時點）

    3. return:
        - Dataset（__len__ 返回 N；__getitem__ 返回 (X_i: [L,F], y_i)）
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

        # 1) 正規化 features：統一到 UTC、排序、去重、捨棄疑似標籤欄
        X = feat_df.copy()
        X.index = _to_utc_index(X.index, assume_tz=assume_tz)
        X = X.sort_index()
        X = X[~X.index.duplicated(keep="last")]
        X = _drop_label_like_cols(X)

        # 特徵欄位選擇
        if feature_cols is None:
            feature_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        else:
            feature_cols = [c for c in feature_cols if c in X.columns]
        if not feature_cols:
            raise ValueError("No valid feature columns selected for EventDataset")

        # 2) 讀取 TBM 事件（t1 欄可選擇性存在；先探測欄位再 parse_dates）
        if not os.path.exists(tbm_csv_path):
            raise FileNotFoundError(f"TBM label CSV not found: {tbm_csv_path}")

        cols_head = pd.read_csv(tbm_csv_path, nrows=0).columns
        parse_cols = ["t0"] + (["t1"] if "t1" in cols_head else [])
        tbm = pd.read_csv(tbm_csv_path, parse_dates=parse_cols)

        req = {"t0", "label", "side"}
        missing = req - set(tbm.columns)
        if missing:
            raise ValueError(f"TBM CSV missing columns: {sorted(missing)}")

        # 轉為 UTC（t0 必有；t1 若不存在則補 NaT[UTC]）
        t0_utc = _to_utc_index(tbm["t0"], assume_tz=assume_tz)
        if "t1" in tbm.columns:
            t1_utc = _to_utc_index(tbm["t1"], assume_tz=assume_tz)
        else:
            t1_utc = pd.Series(pd.DatetimeIndex([pd.NaT] * len(tbm), tz="UTC"), index=tbm.index)
        tbm = tbm.assign(t0_utc=t0_utc, t1_utc=t1_utc)

        # side 正規化：兼容 "1"/"-1" 或 "Long"/"Short"
        if tbm["side"].dtype == object:
            s_num = pd.to_numeric(tbm["side"], errors="coerce")
            if s_num.notna().all():
                tbm["side"] = s_num.astype("Int8")
            else:
                # 大小寫兼容
                mapper = {"long": 1, "short": -1, "Long": 1, "Short": -1,
                          "LONG": 1, "SHORT": -1}
                tbm["side"] = tbm["side"].map(mapper).astype("Int8")
        else:
            tbm["side"] = tbm["side"].astype("Int8")

        # label 容錯轉型（允許字串/缺失）
        tbm["label"] = pd.to_numeric(tbm["label"], errors="coerce")
        tbm = tbm[tbm["label"].notna()].copy()
        tbm["label"] = tbm["label"].astype(int)

        # 方向過濾
        if keep_sides == "long":
            tbm = tbm[tbm["side"] == 1]
        elif keep_sides == "short":
            tbm = tbm[tbm["side"] == -1]

        # 3) t0 對齊到 features 的時間網格，並展開序列視窗
        idx = X.index  # 15m 固定網格（假定）
        L = int(seq_len)
        feats = X.loc[:, feature_cols].astype(np.float32)

        rows: List[EventRow] = []
        seqs: List[np.ndarray] = []
        labels: List[int] = []

        # 掉樣統計
        total_events = 0
        drop_short = 0
        drop_nan = 0
        drop_outside = 0

        # 向量化 searchsorted 找 t0 對齊位置
        t0u = tbm["t0_utc"]
        if align_method == "exact":
            mask_on_grid = t0u.isin(idx)
            t0u = t0u[mask_on_grid]
            tbm = tbm.loc[mask_on_grid]
            pos = idx.get_indexer(t0u)
        elif align_method == "pad":
            pos = idx.searchsorted(t0u, side="right") - 1
            valid = pos >= 0
            pos = pos[valid]
            tbm = tbm.loc[valid]
        else:
            raise ValueError("align_method must be 'exact' or 'pad'")

        # 若提供 allowed_align_index，統一轉 UTC 後再建 set
        allowed_set = None
        if allowed_align_index is not None:
            allowed_set = set(_to_utc_index(allowed_align_index, assume_tz=assume_tz))

        # 逐事件建視窗
        for i, p in enumerate(pos):
            total_events += 1
            if p < L:
                if drop_incomplete:
                    drop_short += 1
                    continue
                else:
                    # 左側 NaN 補齊（並在不完整路徑提供 ffill，以降低 NaN 傳染）
                    seq_window = feats.iloc[0:p].values
                    if len(seq_window) == 0:
                        continue
                    pad = np.full((L - len(seq_window), seq_window.shape[1]), np.nan, dtype=np.float32)
                    arr = np.concatenate([pad, seq_window], axis=0)
                    # 僅在保留不完整視窗時做 ffill（不 bfill）
                    arr = pd.DataFrame(arr).ffill().to_numpy(dtype=np.float32, copy=False)
            else:
                arr = feats.iloc[p - L:p].values  # 嚴格取對齊點之前 L 根（float32）

            # 同時檢查 NaN 與 ±Inf
            if not np.isfinite(arr).all():
                if drop_incomplete:
                    drop_nan += 1
                    continue
                # 保留路徑：此處不再自動處理，交由上游或模型本身

            align_time = idx[p]
            if allowed_set is not None and align_time not in allowed_set:
                drop_outside += 1
                continue

            ev = EventRow(
                t0_raw=pd.Timestamp(tbm.iloc[i]["t0"]),
                t0_utc=pd.Timestamp(tbm.iloc[i]["t0_utc"]),
                t0_align=align_time,
                t1_utc=(pd.Timestamp(tbm.iloc[i]["t1_utc"]) if pd.notna(tbm.iloc[i]["t1_utc"]) else None),
                side=int(tbm.iloc[i]["side"]),
                label=int(tbm.iloc[i]["label"]),
            )

            rows.append(ev)
            seqs.append(arr)                 # feats/Pad 已是 float32，此處不再重複 astype
            labels.append(ev.label)

        if not seqs:
            raise ValueError("No valid events after alignment, history, and NaN/Inf checks.")

        kept = len(seqs)
        print(f"[EventDataset] total={total_events} | kept={kept} | drop_short={drop_short} | drop_nan_or_inf={drop_nan} | drop_outside={drop_outside}")

        # 保存統計供外部存取（避免多進程 print 交錯）
        self.stats = {
            "total": int(total_events),
            "kept": int(kept),
            "drop_short": int(drop_short),
            "drop_nan_or_inf": int(drop_nan),
            "drop_outside": int(drop_outside),
        }

        X_np = np.stack(seqs, axis=0)  # [N, L, F]
        y_np = np.array(labels, dtype=np.int64)  # [N]

        # 預設裝置：優先 CUDA（保留你「預載 GPU」策略）
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.X = torch.tensor(X_np, dtype=torch.float32, device=device)
        self.y = torch.tensor(y_np, dtype=torch.long, device=device)
        self.events: List[EventRow] = rows
        self.feature_cols = feature_cols
        self.seq_len = L
        self.device = device

    def __len__(self) -> int:
        """
        1. 說明: 回傳事件樣本數 N。
        2. inputs: 無
        3. return: int
        """
        return self.y.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1. 說明: 取得第 i 筆樣本 (X_i, y_i)。
        2. inputs:
            - i: int
        3. return:
            - (torch.FloatTensor[L,F], torch.LongTensor[])
        """
        return self.X[i], self.y[i]

    # Convenience accessors
    def event_times(self) -> List[pd.Timestamp]:
        """
        1. 說明: 回傳所有事件的對齊時間（t0_align）。
        2. inputs: 無
        3. return: List[pd.Timestamp]
        """
        return [r.t0_align for r in self.events]

    def event_meta(self) -> List[Dict[str, object]]:
        """
        1. 說明: 回傳事件 meta（原始/UTC t0、對齊時間、t1、side、label）。
        2. inputs: 無
        3. return: List[Dict[str, object]]
        """
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
    """
    1. 說明:
        從 CSV 載入特徵表，並將時間欄轉為「UTC tz-aware」的 DatetimeIndex。
        - 僅移除時間欄，保留其餘欄位；是否為數值由上游自行過濾。
        - 會排序索引並去除重複時間列（保留最後一次出現）。
    2. inputs:
        - path: str
        - time_col: str = "datetime"
        - assume_tz: str = "UTC"
    3. return:
        - pd.DataFrame
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
    cfg: Dict,
    tbm_csv_path: str,
    *,
    seq_len: int = 144,
    keep_sides: Literal["both", "long", "short"] = "both",
    align_method: Literal["exact", "pad"] = "pad",
    device: Optional[str] = None,
) -> EventDataset:
    """
    1. 說明:
        透過 IndicatorLibrary/FeatureComputer 動態計算 15m 特徵，再以 TBM CSV 組成 EventDataset。
        假設 FeatureComputer 內部已完成 anti-leak shift(1)。
    2. inputs:
        - cfg: Dict（需含 data.path / data.index_col / data.freq / features.plan）
        - tbm_csv_path: str
        - seq_len, keep_sides, align_method, device
    3. return:
        - EventDataset
    """
    from ..build_feature_loader.indicators import IndicatorLibrary, FeatureComputer

    raw_path = cfg["data"]["path"]
    index_col = cfg["data"]["index_col"]
    freq = cfg["data"]["freq"]

    # 載入原始 OHLCV
    if str(raw_path).endswith(".csv"):
        raw_df = pd.read_csv(raw_path)
    elif str(raw_path).endswith(".parquet"):
        raw_df = pd.read_parquet(raw_path)
    else:
        raise ValueError("Only .csv or .parquet is supported for data.path")

    lib = IndicatorLibrary(raw_df, freq_check=freq, prefer_time_col=index_col)

    # 不做磁碟快取的 FeatureComputer
    fc = FeatureComputer(lib)

    # 依 plan 計算特徵；假設內部已 shift(1)
    plan = cfg["features"]["plan"]
    feat_df = fc.compute(plan, cfg)

    # 組 Dataset
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
