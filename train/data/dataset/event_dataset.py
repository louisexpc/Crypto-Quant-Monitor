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
import numpy as np
import pandas as pd

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
        本實作以**向量化**方式一次性擷取所有事件視窗，並於 __init__ 產生 tensors。

    2. inputs:
        - feat_df: pd.DataFrame（index=DatetimeIndex）
        - tbm_csv_path: str（TBM CSV 路徑，至少含 t0/label/side）或 tbm_df: DataFrame
        - seq_len: int = 144
        - feature_cols: Optional[List[str]] = None（None 則自動取數值欄）
        - keep_sides: {"both","long","short"} = "both"
        - align_method: {"exact","pad"} = "pad"（exact: t0 必在網格；pad: 對齊到前一根）
        - assume_tz: str = "UTC"（naive 時區假定）
        - drop_incomplete: bool = True（有缺/不足即丟棄；False 則左側 NaN 補齊並 ffill）
        - device: Optional[str] = None（None → 自動選 cuda/cpu）
        - allowed_align_index: Optional[pd.DatetimeIndex] = None（若提供，只保留其內對齊時點）

    3. return:
        - Dataset（__len__ 返回 N；__getitem__ 返回 (X_i: [L,F], y_i)）
    """

    def __init__(
        self,
        feat_df: pd.DataFrame,
        tbm_csv_path: Optional[str] = None,
        tbm_df: Optional[pd.DataFrame] = None,
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
        if tbm_df is None:
            if tbm_csv_path is None:
                raise ValueError("EventDataset requires tbm_df or tbm_csv_path.")
            if not os.path.exists(tbm_csv_path):
                raise FileNotFoundError(f"TBM label CSV not found: {tbm_csv_path}")
            cols_head = pd.read_csv(tbm_csv_path, nrows=0).columns
            parse_cols = ["t0"] + (["t1"] if "t1" in cols_head else [])
            tbm = pd.read_csv(tbm_csv_path, parse_dates=parse_cols)
        else:
            tbm = tbm_df.copy()
            if "t0" not in tbm.columns:
                raise ValueError("tbm_df must contain column 't0'.")
            if "t1" not in tbm.columns:
                tbm["t1"] = pd.NaT
            # 確保 t0/t1 轉為 datetime
            tbm["t0"] = pd.to_datetime(tbm["t0"], errors="coerce", utc=False)
            tbm["t1"] = pd.to_datetime(tbm["t1"], errors="coerce", utc=False)

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
                mapper = {"long": 1, "short": -1, "Long": 1, "Short": -1, "LONG": 1, "SHORT": -1}
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

        # 3) t0 對齊到 features 的時間網格（向量化 searchsorted）
        idx = X.index                      # 15m 固定網格（假定）
        L = int(seq_len)
        feats = X.loc[:, feature_cols].astype(np.float32)
        feats_np = feats.to_numpy(dtype=np.float32, copy=False)

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

        # ---- 允許對齊時點過濾（若提供 allowed_align_index）----
        allowed_mask = np.ones(len(pos), dtype=bool)  # ← 先預設全 True（長度要跟 pos 一樣）
        if allowed_align_index is not None:
            allowed_utc = _to_utc_index(allowed_align_index, assume_tz=assume_tz)
            allowed_set = set(pd.DatetimeIndex(allowed_utc))  # 確保是 UTC tz-aware

            align_times_all = idx[pos]  # 這是 DatetimeIndex（UTC）
            # isin 回傳已是 ndarray(bool)，不同 pandas 版本一致
            allowed_mask = np.asarray(
                pd.DatetimeIndex(align_times_all).isin(pd.DatetimeIndex(list(allowed_set))),
                dtype=bool
            )

        # ---- 長度足夠 (p >= L) ----
        enough_len = pos >= L

        # ---- 統計（以對齊後的母群為基準）----
        total_events = int(len(pos))
        drop_outside = int((~allowed_mask).sum())
        enough_len   = pos >= L
        drop_short   = int((allowed_mask & (~enough_len)).sum())  # 只統計 allowed 範圍內不足 L 的事件
        
        # 最終要 vectorize 的事件（allowed 且長度足夠）
        vec_mask = allowed_mask & enough_len
        pos_vec = pos[vec_mask]
        tbm_vec = tbm.reset_index(drop=True).loc[vec_mask].reset_index(drop=True)

        if len(pos_vec) == 0:
            raise ValueError("No valid events to vectorize after allowed/length checks.")

        # 4) 建立索引矩陣 [N,L] 並一次 gather 出 [N,L,F]
        base = np.arange(L, dtype=np.int64)[None, :]
        positions = (pos_vec[:, None] - L + base)  # [N, L]
        X_vec = feats_np[positions]                 # [N, L, F]

        # 5) 檢查 NaN/Inf；依 drop_incomplete 處置
        finite_mask = np.isfinite(X_vec).all(axis=(1, 2))
        drop_nan = int((~finite_mask).sum())
        if drop_incomplete:
            X_kept = X_vec[finite_mask]
            tbm_kept = tbm_vec.loc[finite_mask].reset_index(drop=True)
        else:
            # 逐樣本沿時間維度做 ffill（避免 batch 維度交互污染）
            X_ff = []
            for i in range(X_vec.shape[0]):
                arr = pd.DataFrame(X_vec[i]).ffill().to_numpy(dtype=np.float32, copy=False)
                X_ff.append(arr)
            X_kept = np.stack(X_ff, axis=0)
            tbm_kept = tbm_vec

        kept = int(X_kept.shape[0])
        if kept == 0:
            raise ValueError("No kept samples after NaN/Inf handling.")

        # 6) 構造 labels 與 EventRow（對齊時間需回推 positions 的右端點 p_i）
        align_times_kept = idx[pos_vec][finite_mask] if drop_incomplete else idx[pos_vec]
        rows: List[EventRow] = []
        for i in range(kept):
            r = EventRow(
                t0_raw=pd.Timestamp(tbm_kept.iloc[i]["t0"]),
                t0_utc=pd.Timestamp(tbm_kept.iloc[i]["t0_utc"]),
                t0_align=pd.Timestamp(align_times_kept[i]),
                t1_utc=(pd.Timestamp(tbm_kept.iloc[i]["t1_utc"]) if pd.notna(tbm_kept.iloc[i]["t1_utc"]) else None),
                side=int(tbm_kept.iloc[i]["side"]),
                label=int(tbm_kept.iloc[i]["label"]),
            )
            rows.append(r)

        y_np = tbm_kept["label"].to_numpy(dtype=np.int64, copy=False)
        X_np = X_kept.astype(np.float32, copy=False)

        # 7) 保存統計（避免多進程 print 交錯）
        self.stats = {
            "total": int(total_events),
            "kept": int(kept),
            "drop_short": int(drop_short),
            "drop_nan_or_inf": int(drop_nan),
            "drop_outside": int(drop_outside),
        }
        print(f"[EventDataset] total={total_events} | kept={kept} | "
              f"drop_short={drop_short} | drop_nan_or_inf={drop_nan} | drop_outside={drop_outside}")

        # 8) 張量化與裝置
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.X = torch.tensor(X_np, dtype=torch.float32, device=device)   # [N,L,F]
        self.y = torch.tensor(y_np, dtype=torch.long, device=device)      # [N]
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
