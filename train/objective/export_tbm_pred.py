# export_tbm_pred.py
from __future__ import annotations
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import amp
import sys, os

# 專案內部匯入
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


# ---------------- main export ----------------
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
    decision_mode: str = "mean_prob",   # "mean_prob" | "fold_vote" | "both"
) -> str:
    """
    1. 說明:
       以某個 trial 的各 fold 最佳模型，對 [date_start,date_end] 區間的事件做推論並回寫 CSV。
       - 僅對 (keep_sides, 日期遮罩) 的子集做推論，但輸出 CSV 保留全部 rows。
       - 決策模式：
         * 'mean_prob'：各 fold 機率平均 → 用全域門檻判斷（soft voting）。
         * 'fold_vote'：各 fold 用各自 best_val_thresh 先產生 0/1 → 多數決（hard voting）。
         * 'both'：同時輸出兩組欄位：{pred}_prob* 與 {pred}_vote*。
       - 產出欄位：
         * soft 版：{pred}_prob（0/1）、{pred}_prob_p0、{pred}_prob_p1、{pred}_prob_thr
         * hard 版：{pred}_vote（0/1）、{pred}_vote_votes_0、{pred}_vote_votes_1
           （另附 {pred}_vote_p0、{pred}_vote_p1 供參考，以及健檢 {pred}_vote_votes_total、{pred}_vote_margin）
    2. inputs:
       - cfg: Dict                 統一配置（資料/特徵/序列/scaler/AMP/裝置等）
       - df_index: DatetimeIndex   原始全域時間索引（決定 fold split 的索引基準）
       - folds: List[Dict]         每個 fold 的 split 定義（傳給 split_fold_to_indices）
       - fold_models: List[Tuple(model, fold_dict, result)]  模型與對應 fold 訓練結果（含 best_val_thresh/temperature）
       - date_start/date_end: str  推論期間（將轉為 UTC）
       - src_tbm_csv_path: str|None  TBM 來源 CSV（不給則用 cfg["label"]["tbm_csv_path"]）
       - save_to_path: str|None    輸出 CSV 路徑（不給則用來源檔名 + 區間字尾）
       - output_column: str        欄位基名（預設 "pred"）
       - threshold_override: float|None  覆寫 soft voting 的全域門檻（預設用各 fold 門檻均值）
       - decision_mode: str        "mean_prob" | "fold_vote" | "both"
    3. return:
       - str: 實際寫出的 CSV 路徑
    """
    assert cfg["task"]["type"] == "classification", "目前僅支援分類任務的匯出"

    device = "cuda" if (torch.cuda.is_available() and str(cfg.get("device", "cuda")) == "cuda") else "cpu"

    # AMP 選擇
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

    # ---- features ----
    feat_df = _load_precomputed_features(str(cfg["data"]["path"]))
    feat_cols = select_plan_columns(feat_df, cfg)
    feat_cols = [c for c in feat_cols if c in feat_df.columns]
    feat_df = feat_df.loc[:, feat_cols].astype(np.float32)
    idx_all = pd.DatetimeIndex(feat_df.index)  # UTC, sorted

    # ---- TBM (完整表，保留為輸出底稿) ----
    tbm_path = str(src_tbm_csv_path) if (src_tbm_csv_path not in (None, "", "None", "none", "NULL", "null")) else str(cfg["label"]["tbm_csv_path"])
    if not os.path.exists(tbm_path):
        raise FileNotFoundError(f"TBM label CSV not found: {tbm_path}")

    tbm_all = pd.read_csv(tbm_path, parse_dates=["t0"])
    if "__rid" not in tbm_all.columns:
        tbm_all["__rid"] = np.arange(len(tbm_all), dtype=np.int64)

    assume_tz = str(cfg.get("data", {}).get("assume_tz", "UTC"))
    t0u_all = _to_utc_index(tbm_all["t0"], assume_tz=assume_tz)

    # ---- side / date 遮罩 → 僅供推論的子集（不改 tbm_all！）----
    keep_sides = str(cfg["label"].get("keep_sides", "both")).lower()

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

    side_i = tbm_all["side"].map(_side_to_int)
    if keep_sides == "long":
        mask_side = side_i.eq(1)
    elif keep_sides == "short":
        mask_side = side_i.eq(-1)
    else:
        mask_side = side_i.isin([1, -1])  # both

    mask_date = (t0u_all >= pd.Timestamp(date_start, tz="UTC")) & (t0u_all <= pd.Timestamp(date_end, tz="UTC"))

    tbm_sel = tbm_all.loc[mask_side & mask_date].copy()
    t0u_sel = t0u_all[mask_side & mask_date]
    if tbm_sel.empty:
        raise ValueError("指定期間內（且符合 side 遮罩）的 TBM 事件為空，無法推論匯出。")

    # ---- 對齊時間（僅為推論候選）----
    align_method = str(cfg.get("label", {}).get("align_method", "pad")).lower()
    allowed_align = _align_times(t0u_sel, idx_all, align_method)

    # === 準備 rid 索引（靜態，不會被消耗） ===
    tbm_sel = tbm_sel.copy()
    tbm_sel["t0u"] = _to_utc_index(tbm_sel["t0"], assume_tz=assume_tz)
    events_order = tbm_sel.sort_values(["t0u", "__rid"])

    # key: t0_utc -> value: 依 __rid 順序的 rid 列表（保持穩定對應）
    rid_lists_by_t0: Dict[pd.Timestamp, List[int]] = {}
    for rid, t0u in zip(events_order["__rid"].to_numpy(), events_order["t0u"].to_numpy()):
        rid_lists_by_t0.setdefault(t0u, []).append(int(rid))

    # ---- dataloader / 推論通用設定 ----
    bs = int(((cfg.get("post_infer", {}) or {}).get("tbm_concat", {}) or {}).get("batch_size", cfg["train"]["batch_size"]))
    L = int(cfg["sequence"]["seq_len"])
    pos_map = pd.Series(np.arange(len(idx_all)), index=idx_all)

    # 先蒐集所有 fold 的機率與門檻
    per_fold_prob: Dict[int, Dict[int, float]] = {}  # fold_idx -> {rid: p1}
    thr_per_fold: Dict[int, float] = {}              # 修正：以 fold_idx 為 key 收集門檻，避免錯位

    for fold_idx, (model, fold_d, result) in enumerate(fold_models):
        # 每個 fold 自己的 t0 配對指標（不互相影響）
        rid_counter = defaultdict(int)  # key: t0_utc -> 已分配次數（當作 rid_lists_by_t0 的索引）

        # 1) split：決定 scaler 擬合窗口（train）
        df_idx = pd.DataFrame(index=pd.DatetimeIndex(df_index))
        tr_idx, va_idx, te_idx = split_fold_to_indices(df_idx, fold_d, cfg)
        train_align = _align_times(tr_idx, idx_all, align_method)

        fit_pos: List[int] = []
        for at in train_align:
            p = int(pos_map.get(at, -1))
            if p <= 0:
                continue
            start = max(0, p - L)
            fit_pos.extend(range(start, p))
        fit_pos = np.unique(np.array(fit_pos, dtype=int))
        fit_index = idx_all[fit_pos] if len(fit_pos) else train_align

        # 2) scaler 擬合 + transform
        cols_to_scale = pick_cols_to_scale(feat_df.loc[fit_index, feat_cols], feat_cols)
        scaler_kind = cfg["sequence"]["scaler"]
        min_frac = float(cfg["sequence"].get("min_frac", 0.2))
        scaler = _get_scaler(scaler_kind, window=L, min_frac=min_frac)

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

        # 3) 建推論 Dataset（僅含 allowed_align 事件）
        ds_inf = EventDataset(
            feat_scaled,
            tbm_path,
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
            pin_memory=False,
        )

        # 4) 溫度與門檻（收集）
        best_T = float(result.get("temperature", 1.0) or 1.0)
        best_thr = result.get("best_val_thresh", None)
        if best_thr is not None:
            thr_per_fold[fold_idx] = float(best_thr)

        # 5) 推論
        per_fold_prob[fold_idx] = {}
        model = model.to(device)
        model.eval()
        with torch.no_grad(), amp.autocast(
            device_type=("cuda" if device == "cuda" else "cpu"),
            dtype=dtype,
            enabled=(dtype is not None and device == "cuda"),
        ):
            ptr = 0
            for Xb, _ in dl:
                Xb = Xb.to(device, non_blocking=False)
                logits = model(Xb) / best_T

                if logits.ndim == 1 or logits.shape[-1] == 1:
                    p1 = torch.sigmoid(logits.float().squeeze(-1))
                else:
                    p1 = torch.softmax(logits.float(), dim=1)[:, 1]

                p1_np = p1.detach().to("cpu").numpy().astype(np.float32)

                for k in range(len(p1_np)):
                    ev = ds_inf.events[ptr + k]
                    key_t0 = getattr(ev, "t0_utc", None)
                    if key_t0 is None:
                        # 保險：若 Dataset 無 t0_utc，嘗試 ev.t0 或 ev.time
                        key_t0 = getattr(ev, "t0", getattr(ev, "time", None))
                        key_t0 = _to_utc_index(pd.Index([key_t0]))[0]

                    # A 方案：每 fold 自己的計數器，不消耗共享列表
                    i = rid_counter[key_t0]                 # 這個 t0 已經分配到第幾個重複事件
                    rid_list = rid_lists_by_t0.get(key_t0, [])
                    if i >= len(rid_list):
                        # 這個 t0 在 tbm_sel 中沒有第 i 個重複（可能該事件被過濾/無法建序列），略過
                        continue

                    rid = rid_list[i]                       # 取第 i 個 __rid（不消耗原列表）
                    rid_counter[key_t0] += 1                # 下次相同 t0 就配到下一個 __rid

                    per_fold_prob[fold_idx][rid] = float(p1_np[k])
                ptr += len(p1_np)

        # 每個 fold 推論完釋放 GPU 記憶體
        try:
            model = model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # === 聚合決策（同時支援 prob 與 vote）===
    dm = str(decision_mode).lower()
    base = str(output_column)

    # soft/mean 版欄位
    col_prob_pred = f"{base}_prob"
    col_prob_p0   = f"{base}_prob_p0"
    col_prob_p1   = f"{base}_prob_p1"
    col_prob_thr  = f"{base}_prob_thr"

    # hard/vote 版欄位
    col_vote_pred   = f"{base}_vote"
    col_vote_p0     = f"{base}_vote_p0"   # 參考：平均機率
    col_vote_p1     = f"{base}_vote_p1"   # 參考：平均機率
    col_vote_v0     = f"{base}_vote_votes_0"
    col_vote_v1     = f"{base}_vote_votes_1"
    col_vote_total  = f"{base}_vote_votes_total"   # 健檢：有效投票數
    col_vote_margin = f"{base}_vote_margin"        # 健檢：勝出邊際 (v1 - v0)

    # 找到所有有任一 fold 機率的 rid（有效推論事件）
    rid_union = sorted(set().union(*[set(d.keys()) for d in per_fold_prob.values()])) if per_fold_prob else []

    # 全域門檻（soft 用），以各 fold best_val_thresh 平均（或覆寫）
    thr_vals = list(thr_per_fold.values())
    thr_mean = float(np.mean(thr_vals)) if thr_vals else 0.5
    thr_used_global = float(threshold_override) if (threshold_override is not None) else thr_mean

    # 逐 fold 門檻（hard 用），若該 fold 無門檻資訊則退回全域門檻
    thr_map: Dict[int, float] = {i: thr_per_fold.get(i, thr_used_global) for i, _ in enumerate(fold_models)}

    # 蒐集兩組結果
    prob_records: List[Tuple[int, float, float, float, int]] = []  # rid, p0, p1, thr, yhat_prob
    vote_records: List[Tuple[int, float, float, int, int, int, int, int]] = []  # rid, p0, p1, v0, v1, yhat_vote, total, margin

    for rid in rid_union:
        # 收集此事件在所有 fold 的機率
        probs = [per_fold_prob[f][rid] for f in per_fold_prob if rid in per_fold_prob[f]]
        if not probs:
            continue

        p1_mean = float(np.mean(probs))
        p0_mean = float(1.0 - p1_mean)

        # soft/mean 機率 → 單一門檻
        if dm in {"mean_prob", "both"}:
            yhat_prob = int(p1_mean >= thr_used_global)
            prob_records.append((rid, p0_mean, p1_mean, thr_used_global, yhat_prob))

        # hard/多數決（各 fold 自門檻 → 投票）
        if dm in {"fold_vote", "both"}:
            v1 = 0
            v0 = 0
            for f_idx, d in per_fold_prob.items():
                if rid not in d:
                    continue
                p1 = float(d[rid])
                f_thr = float(thr_map.get(f_idx, thr_used_global))
                yhat_f = int(p1 >= f_thr)
                if yhat_f == 1:
                    v1 += 1
                else:
                    v0 += 1
            total = v0 + v1
            margin = v1 - v0
            yhat_vote = 1 if (v1 > v0) else 0
            vote_records.append((rid, p0_mean, p1_mean, v0, v1, yhat_vote, total, margin))

    # 組 DataFrame
    out_df = tbm_all.copy()
    if dm in {"mean_prob", "both"} and prob_records:
        prob_df = pd.DataFrame(prob_records, columns=["__rid", col_prob_p0, col_prob_p1, col_prob_thr, col_prob_pred])
        out_df = out_df.merge(prob_df, on="__rid", how="left", validate="one_to_one")

    if dm in {"fold_vote", "both"} and vote_records:
        vote_df = pd.DataFrame(
            vote_records,
            columns=["__rid", col_vote_p0, col_vote_p1, col_vote_v0, col_vote_v1, col_vote_pred, col_vote_total, col_vote_margin],
        )
        out_df = out_df.merge(vote_df, on="__rid", how="left", validate="one_to_one")

    # 欄位相對順序調整：各群內維持 p0→p1→thr/votes→pred（僅把這群移到表尾，其他順序不動）
    def _move_group_to_tail(df: pd.DataFrame, cols_in_order: List[str]) -> pd.DataFrame:
        """
        1. 說明:
            將指定欄位群（若存在）移到 DataFrame 末尾，並保留其相對順序。
        2. inputs:
            - df: pd.DataFrame
            - cols_in_order: 欲搬移的欄位（相對順序）
        3. return:
            - pd.DataFrame（新順序視圖）
        """
        exist = [c for c in cols_in_order if c in df.columns]
        if not exist:
            return df
        base_cols = [c for c in df.columns if c not in exist]
        return df[base_cols + exist]

    if dm in {"mean_prob", "both"}:
        out_df = _move_group_to_tail(out_df, [col_prob_p0, col_prob_p1, col_prob_thr, col_prob_pred])
    if dm in {"fold_vote", "both"}:
        out_df = _move_group_to_tail(out_df, [col_vote_p0, col_vote_p1, col_vote_v0, col_vote_v1, col_vote_total, col_vote_margin, col_vote_pred])

    # ---- 輸出 ----
    if save_to_path is None:
        base_name = os.path.basename(str(cfg["label"]["tbm_csv_path"] if src_tbm_csv_path in (None, "", "None", "none", "NULL", "null") else src_tbm_csv_path))
        name, ext = os.path.splitext(base_name)
        s = date_start.replace("-", "")
        e = date_end.replace("-", "")
        save_to_path = os.path.join(os.path.dirname(str(cfg["label"]["tbm_csv_path"] if src_tbm_csv_path in (None, "", "None", "none", "NULL", "null") else src_tbm_csv_path)),
                                    f"{name}_with_{output_column}_{s}_{e}{ext}")
    os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
    out_df.to_csv(save_to_path, index=False)

    # ---- summary ----（分別對 soft / hard 輸出統計）
    try:
        import json
        summary = {
            "date_start": date_start,
            "date_end": date_end,
            "assume_tz": str(cfg.get("data", {}).get("assume_tz", "UTC")),
            "keep_sides": str(cfg["label"].get("keep_sides", "both")).lower(),
            "align_method": str(cfg.get("label", {}).get("align_method", "pad")).lower(),
            "seq_len": int(cfg["sequence"]["seq_len"]),
            "decision_mode": dm,
            "thr_mean": float(thr_mean),
            "thr_used_global": (float(thr_used_global) if dm in {"mean_prob", "both"} else None),
            # 原始每 fold 訓練得到的門檻（可能缺）
            "thr_list_per_fold_raw": {int(k): float(v) for k, v in thr_per_fold.items()},
            # 實際投票時使用的每 fold 門檻（缺的用全域門檻補）
            "thr_used_per_fold": {int(i): float(thr_map[i]) for i, _ in enumerate(fold_models)},
            "rows_in_output": int(len(out_df)),
            "events_predicted": int(out_df["__rid"].isin(rid_union).sum()),
        }
        mask_out = out_df["__rid"].isin(rid_union)

        if dm in {"mean_prob", "both"}:
            ps = out_df.loc[mask_out, col_prob_p1].astype("float64")
            summary["soft_prob"] = {
                "pred_ones": int((out_df.loc[mask_out, col_prob_pred] == 1).sum()) if col_prob_pred in out_df.columns else None,
                "pred_zeros": int((out_df.loc[mask_out, col_prob_pred] == 0).sum()) if col_prob_pred in out_df.columns else None,
                "prob_mean": float(ps.mean(skipna=True)) if len(ps) else None,
                "prob_min": float(ps.min(skipna=True)) if len(ps) else None,
                "prob_max": float(ps.max(skipna=True)) if len(ps) else None,
            }
        if dm in {"fold_vote", "both"}:
            ps = out_df.loc[mask_out, col_vote_p1].astype("float64")
            summary["hard_vote"] = {
                "pred_ones": int((out_df.loc[mask_out, col_vote_pred] == 1).sum()) if col_vote_pred in out_df.columns else None,
                "pred_zeros": int((out_df.loc[mask_out, col_vote_pred] == 0).sum()) if col_vote_pred in out_df.columns else None,
                "prob_mean": float(ps.mean(skipna=True)) if len(ps) else None,
                "prob_min": float(ps.min(skipna=True)) if len(ps) else None,
                "prob_max": float(ps.max(skipna=True)) if len(ps) else None,
                "votes_total_mean": float(out_df.loc[mask_out, col_vote_total].mean(skipna=True)) if col_vote_total in out_df.columns else None,
                "votes_total_min": int(out_df.loc[mask_out, col_vote_total].min(skipna=True)) if col_vote_total in out_df.columns else None,
                "votes_total_max": int(out_df.loc[mask_out, col_vote_total].max(skipna=True)) if col_vote_total in out_df.columns else None,
            }

        with open(os.path.splitext(save_to_path)[0] + ".summary.json", "w", encoding="utf-8") as jf:
            json.dump(summary, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return save_to_path
