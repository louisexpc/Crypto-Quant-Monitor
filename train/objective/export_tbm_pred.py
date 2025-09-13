# export_tbm_pred.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import amp
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from build_feature_loader.dataloader import (
    select_plan_columns,
    split_fold_to_indices,
)
from build_feature_loader.scalar import _get_scaler, ColumnSubsetScaler, pick_cols_to_scale
from dataset.event_dataset import EventDataset

def _to_utc_index(idx: pd.Index | pd.Series, assume_tz: str = "UTC") -> pd.DatetimeIndex:
    """
    1. 說明:
        將任意時間索引/序列正規化為 tz-aware 的 UTC DatetimeIndex。
        - naive → 先以 assume_tz localize，再轉 UTC
        - 已帶時區 → 直接轉 UTC
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


def _load_precomputed_features(pre_path: str) -> pd.DataFrame:
    """
    1. 說明:
        載入預算特徵（csv/parquet），建立 UTC DatetimeIndex（來自 'datetime' 或 'timestamp' 欄），
        排序並去重（保留最後一次）。
    2. inputs:
        - pre_path: str  路徑（.csv 或 .parquet）
    3. return:
        - pd.DataFrame  (index=UTC DatetimeIndex)
    """
    if pre_path.endswith(".csv"):
        df = pd.read_csv(pre_path)
    elif pre_path.endswith(".parquet"):
        df = pd.read_parquet(pre_path)
    else:
        raise ValueError("features.precomputed.path 只支援 .csv 或 .parquet")

    # 設定索引：datetime / timestamp / 已有 DatetimeIndex
    if "datetime" in df.columns:
        idx = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.drop(columns=["datetime"])
        df.index = idx
    elif "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        ts_nonan = ts.dropna()
        if len(ts_nonan) == 0:
            raise ValueError("timestamp 欄全為 NaN")
        q = float(ts_nonan.quantile(0.5))
        unit = "ms" if q > 1_000_000_000_000 else "s"
        idx = pd.to_datetime(ts, unit=unit, utc=True)
        df = df.drop(columns=["timestamp"])
        df.index = idx
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = _to_utc_index(df.index)
    else:
        raise TypeError("預算特徵檔需包含 'datetime' 或 'timestamp' 欄位，或已是 DatetimeIndex")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _align_times(t0_index: pd.DatetimeIndex, idx_all: pd.DatetimeIndex, method: str) -> pd.DatetimeIndex:
    """
    1. 說明:
        將一組時間 t0 對齊到 idx_all（固定網格）的時間點。
        - exact: t0 必須在網格上
        - pad  : 對齊到 t0 前一根（floor）
    2. inputs:
        - t0_index: pd.DatetimeIndex
        - idx_all: pd.DatetimeIndex（網格，需為 UTC）
        - method: str in {"exact","pad"}
    3. return:
        - pd.DatetimeIndex（對齊到 idx_all 的時間）
    """
    method = str(method).lower()
    t0u = _to_utc_index(t0_index, assume_tz="UTC")
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


def export_tbm_predictions_for_trial(
    *,
    cfg: Dict,
    df_index: pd.DatetimeIndex,
    folds: List[Dict],
    fold_models: List[Tuple[torch.nn.Module, Dict, Dict]],  # (model, fold_dict, result)
    date_start: str,
    date_end: str,
    src_tbm_csv_path: str | None = None,
    save_to_path: str | None = None,
    output_column: str = "pred",
    threshold_override: float | None = None,
) -> str:
    """
    1. 說明:
        以「單次 trial 的各 fold 最佳模型」對指定期間 [date_start, date_end] 的 TBM 事件做推論，
        將「各 fold 正類機率平均」後，以「平均門檻/覆蓋門檻」產生二值預測，並把
        - {output_column}     : 0/1 預測
        - {output_column}_p1  : 正類機率（成功/TP 機率）
        - {output_column}_p0  : 負類機率（失敗/SL 機率 = 1 - p1）
        寫回到 TBM CSV 的副本。

        注意：以「原始 t0_utc」做 key（避免 pad 對齊時多事件撞在同一根）。
    2. inputs:
        - cfg, df_index, folds, fold_models, date_start, date_end, src_tbm_csv_path,
          save_to_path, output_column, threshold_override
    3. return:
        - str: 實際輸出的 CSV 路徑
    """


    assert cfg["task"]["type"] == "classification", "目前僅支援分類任務的匯出"

    device = "cuda" if (torch.cuda.is_available() and str(cfg.get("device", "cuda")) == "cuda") else "cpu"

    # AMP dtype 自動：CUDA 優先 bf16（若支援）否則 fp16；CPU 停用
    dtype = None
    if bool(cfg["train"].get("amp", True)):
        kind = str(cfg["train"].get("amp_dtype", "auto")).lower()
        if kind in {"fp16", "float16"}:
            dtype = torch.float16
        elif kind in {"bf16", "bfloat16"}:
            dtype = torch.bfloat16
        else:
            if device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = None

    # --- 時區假定（避免把 naive t0 誤當 UTC） ---
    assume_tz = str(cfg.get("data", {}).get("assume_tz", "UTC"))

    # 1) 載入 precomputed 特徵 + 欄位挑選
    feat_df = _load_precomputed_features(str(cfg["data"]["path"]))
    feat_cols = select_plan_columns(feat_df, cfg)
    feat_cols = [c for c in feat_cols if c in feat_df.columns]
    feat_df = feat_df.loc[:, feat_cols].astype(np.float32)

    # 2) 讀取 TBM CSV，篩選日期區間 + keep_sides
    tbm_path = str(src_tbm_csv_path) if (src_tbm_csv_path not in (None, "", "None", "none", "NULL", "null")) else str(cfg["label"]["tbm_csv_path"])
    if not os.path.exists(tbm_path):
        raise FileNotFoundError(f"TBM label CSV not found: {tbm_path}")

    tbm = pd.read_csv(tbm_path, parse_dates=["t0"])

    keep_sides = str(cfg["label"].get("keep_sides", "both")).lower()
    if keep_sides in {"long", "short"}:
        def _side_to_int(x):
            s = str(x).strip().lower()
            if s in {"long", "buy", "l", "+1"}:
                return 1
            if s in {"short", "sell", "s", "-1"}:
                return -1
            try:
                return int(float(s))
            except Exception:
                return np.nan
        side_i = tbm["side"].map(_side_to_int)
        tbm = tbm[side_i == (1 if keep_sides == "long" else -1)]

    # 時間範圍（以 t0_utc 比對）
    t0u_all = _to_utc_index(tbm["t0"], assume_tz=assume_tz)
    mask = (t0u_all >= pd.Timestamp(date_start, tz="UTC")) & (t0u_all <= pd.Timestamp(date_end, tz="UTC"))
    tbm_sel = tbm.loc[mask].copy()
    # DatetimeIndex 沒有 .loc，改用布林切片
    t0u_sel = t0u_all[mask]
    if tbm_sel.empty:
        raise ValueError("指定期間內 TBM 事件為空，無法推論匯出。")

    # 3) 準備允許的對齊時間（期間內所有事件）
    idx_all = pd.DatetimeIndex(feat_df.index)  # 已為 UTC
    align_method = str(cfg.get("label", {}).get("align_method", "pad")).lower()
    allowed_align = _align_times(t0u_sel, idx_all, align_method)

    # --- 推論設定 ---
    bs = int(cfg["train"]["batch_size"])

    # 5) 逐 fold 推論：建立 scaler（依該 fold 訓練窗口），對整表轉換後重建 Dataset（CPU），batch 推論到 GPU
    prob_acc: Dict[pd.Timestamp, List[float]] = {}  # key = 原始 t0_utc
    thr_list: List[float] = []
    L = int(cfg["sequence"]["seq_len"])

    for (model, fold_d, result) in fold_models:
        # ---- 切 train/val/test，取 train 組 scaler 擬合窗口 ----
        df_idx = pd.DataFrame(index=pd.DatetimeIndex(df_index))
        tr_idx, va_idx, te_idx = split_fold_to_indices(df_idx, fold_d, cfg)

        train_align = _align_times(tr_idx, idx_all, align_method)
        pos_map = pd.Series(np.arange(len(idx_all)), index=idx_all)
        fit_pos: List[int] = []
        for at in train_align:
            p = int(pos_map.get(at, -1))
            if p <= 0:
                continue
            start = max(0, p - L)
            fit_pos.extend(range(start, p))
        fit_pos = np.unique(np.array(fit_pos, dtype=int))
        fit_index = idx_all[fit_pos] if len(fit_pos) else train_align

        from build_feature_loader.scalar import _get_scaler, ColumnSubsetScaler, pick_cols_to_scale
        cols_to_scale = pick_cols_to_scale(feat_df.loc[fit_index, feat_cols], feat_cols)
        scaler_kind = cfg["sequence"]["scaler"]
        min_frac = float(cfg["sequence"].get("min_frac", 0.2))
        scaler = _get_scaler(scaler_kind, window=L, min_frac=min_frac)

        # Timesafe scaler：整表 transform；否則 ColumnSubsetScaler 在 fit_index 擬合後 transform 全表
        if hasattr(scaler, "is_timesafe") and getattr(scaler, "is_timesafe", False):
            feat_scaled = scaler.transform_full(feat_df, cols_to_scale=cols_to_scale)
        else:
            if scaler is None:
                feat_scaled = feat_df
            else:
                sk = ColumnSubsetScaler(scaler, all_cols=feat_cols, cols_to_scale=cols_to_scale)
                sk.fit_df(feat_df.loc[fit_index, feat_cols])
                arr = feat_df.loc[:, feat_cols].values.astype(np.float32, copy=False)
                arr = sk.transform(arr)
                feat_scaled = feat_df.copy()
                feat_scaled.loc[:, feat_cols] = arr

        ds_inf = EventDataset(
            feat_scaled, tbm_path,
            seq_len=L,
            feature_cols=feat_cols,
            keep_sides=keep_sides,
            align_method=align_method,
            device="cpu",
            allowed_align_index=allowed_align,
        )
        dl = DataLoader(
            ds_inf,
            batch_size=bs,
            shuffle=False,
            drop_last=False,
            num_workers=0,           
            pin_memory=False          
        )

        # 溫度與門檻
        best_T = float(result.get("temperature", 1.0) or 1.0)
        best_thr = result.get("best_val_thresh", None)
        if best_thr is not None:
            thr_list.append(float(best_thr))

        # 推論
        model = model.to(device)
        model.eval()
        with torch.no_grad(), amp.autocast(
            device_type=("cuda" if device == "cuda" else "cpu"),
            dtype=dtype,
            enabled=(dtype is not None and device == "cuda"),
        ):
            ptr = 0
            for Xb, _ in dl:
                # 將 batch 搬到與模型相同的裝置，避免 CPU/GPU 混用導致 index_select 錯誤
                Xb = Xb.to(device, non_blocking=False)
                logits = model(Xb) / best_T

                # 單 logit → sigmoid；雙 logit → softmax 取第 2 類
                if logits.ndim == 1 or logits.shape[-1] == 1:
                    p = torch.sigmoid(logits.float().squeeze(-1))
                else:
                    p = torch.softmax(logits.float(), dim=1)[:, 1]

                p_pos = p.detach().to("cpu").numpy().astype(np.float32)

                # 依序對應 ds_inf.events，用「原始 t0_utc」當 key
                for k in range(len(p_pos)):
                    ev = ds_inf.events[ptr + k]
                    key = ev.t0_utc
                    prob_acc.setdefault(key, []).append(float(p_pos[k]))
                ptr += len(p_pos)

    # 6) 機率平均 + 門檻
    mean_thr = float(np.mean(thr_list)) if thr_list else 0.5
    thr_used = float(threshold_override) if (threshold_override is not None) else mean_thr
    pred_map: Dict[pd.Timestamp, int] = {}
    prob_map: Dict[pd.Timestamp, float] = {}
    for key_t0, vals in prob_acc.items():
        p = float(np.mean(vals))
        prob_map[key_t0] = p
        pred_map[key_t0] = int(p >= thr_used)

    # 7) 寫回 TBM CSV（以原始 t0_utc 對應）
    # 取得期間內事件的 t0_utc → 生成對應的機率與預測
    prob_series = []
    pred_series = []
    for t0 in t0u_sel:
        p = prob_map.get(t0, np.nan)
        prob_series.append(p)
        pred_series.append(int(p >= thr_used) if pd.notna(p) else np.nan)

    tbm_sel = tbm_sel.copy()
    col_pred = output_column
    col_p1 = f"{output_column}_p1"
    col_p0 = f"{output_column}_p0"
    tbm_sel[col_pred] = pred_series
    tbm_sel[col_p1] = prob_series
    tbm_sel[col_p0] = [float(1.0 - p) if pd.notna(p) else np.nan for p in prob_series]

    # merge 回原 CSV：偵測欄位後再 parse，並以 t0_key（UTC 字串）做鍵
    # cols_head_all = pd.read_csv(tbm_path, nrows=0).columns
    # parse_cols_all = ["t0"] + (["t1"] if "t1" in cols_head_all else [])
    tbm_all = pd.read_csv(tbm_path, parse_dates=["t0"])

    tbm_all["t0_key"] = _to_utc_index(tbm_all["t0"], assume_tz=assume_tz).astype(str)
    tbm_sel["t0_key"]  = _to_utc_index(tbm_sel["t0"],  assume_tz=assume_tz).astype(str)

    merge_cols = ["t0_key", col_pred, col_p1, col_p0]
    tbm_all = tbm_all.merge(tbm_sel[merge_cols], on="t0_key", how="left", suffixes=("", "_new"))

    # 逐欄處理 _new 覆蓋/改名
    for col in [col_pred, col_p1, col_p0]:
        new_col = f"{col}_new"
        if new_col in tbm_all.columns and col in tbm_all.columns:
            msk = tbm_all[new_col].notna()
            tbm_all.loc[msk, col] = tbm_all.loc[msk, new_col]
            tbm_all = tbm_all.drop(columns=[new_col])
        elif new_col in tbm_all.columns and col not in tbm_all.columns:
            tbm_all = tbm_all.rename(columns={new_col: col})
    tbm_all = tbm_all.drop(columns=["t0_key"])

    # 8) 輸出到指定路徑（未指定則在原檔旁建立副本）
    if save_to_path is None:
        base = os.path.basename(tbm_path)
        name, ext = os.path.splitext(base)
        s = date_start.replace("-", "")
        e = date_end.replace("-", "")
        save_to_path = os.path.join(os.path.dirname(tbm_path), f"{name}_with_{output_column}_{s}_{e}{ext}")
    os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
    tbm_all.to_csv(save_to_path, index=False)

    # 9) 輸出簡易摘要（json）
    try:
        import json
        ps = pd.Series(prob_series, dtype="float64")
        n_total = int(len(tbm_sel))
        n_filled = int(ps.notna().sum())
        n_ones = int(pd.Series(pred_series).fillna(-1).eq(1).sum())
        n_zeros = int(pd.Series(pred_series).fillna(-1).eq(0).sum())
        summary = {
            "date_start": date_start,
            "date_end": date_end,
            "assume_tz": assume_tz,
            "keep_sides": keep_sides,
            "align_method": align_method,
            "seq_len": int(cfg["sequence"]["seq_len"]),
            "thr_used": float(thr_used),
            "thr_mean": float(mean_thr),
            "thr_list_per_fold": [float(x) for x in thr_list],
            "events_in_range": n_total,
            "events_predicted": n_filled,
            "pred_zeros": n_zeros,
            "pred_ones": n_ones,
            "prob_mean": float(ps.mean(skipna=True)) if n_filled else None,
            "prob_min": float(ps.min(skipna=True)) if n_filled else None,
            "prob_max": float(ps.max(skipna=True)) if n_filled else None,
        }
        with open(os.path.splitext(save_to_path)[0] + ".summary.json", "w", encoding="utf-8") as jf:
            json.dump(summary, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return save_to_path
