# export_tbm_pred.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import amp
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, confusion_matrix

# 專案內部匯入
from train.data.folds import split_fold_to_indices
from train.data.scalers import _get_scaler, ColumnSubsetScaler, pick_cols_to_scale
from train.data.dataset.event_dataset import EventDataset
from train.data.dataloaders.base import load_precomputed_features, align_times, ensure_utc_index
from train.models.model_factory import build_model


def _resolve_checkpoint_path(model_ref: Any, trial_dir: Optional[Path], fold_idx: int) -> Optional[Path]:
    """Return absolute checkpoint path if model_ref points to a saved file."""
    if isinstance(model_ref, dict):
        model_ref = model_ref.get("path")
    if model_ref is None:
        return None
    if isinstance(model_ref, (str, os.PathLike, Path)):
        path = Path(model_ref)
        if not path.is_absolute() and trial_dir is not None:
            path = (trial_dir / path).resolve()
        if not path.exists() and trial_dir is not None:
            tail = Path(*path.parts[-2:]) if len(path.parts) >= 2 else Path(path.name)
            alt = trial_dir / tail
            if alt.exists():
                path = alt
        if not path.exists():
            raise FileNotFoundError(f"[tbm_exporter] fold {fold_idx} checkpoint not found: {path}")
        return path
    return None


def _load_model_from_checkpoint(
    *,
    checkpoint_path: Path,
    cfg: Dict,
    feat_df: pd.DataFrame,
    default_cols: List[str],
    result: Dict,
    fold_idx: int,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    ckpt_cols = ckpt.get("feature_columns") or result.get("_feature_columns")
    feat_cols_current = default_cols
    if ckpt_cols:
        feat_cols_current = [c for c in ckpt_cols if c in feat_df.columns]
        if not feat_cols_current:
            raise ValueError(f"[tbm_exporter] fold {fold_idx} checkpoint columns missing in feature DF")
    model = build_model(cfg, len(feat_cols_current), feat_cols_current)
    model.load_state_dict(state_dict)
    return model, feat_cols_current

def collapse_mask(result: dict) -> bool:
    """
    1. 說明:
        避免 train 壞的模型參與投票
        規則1: val_loss_f > val_loss_0
        規則2: pos_ratio > 0.8 或 < 0.2
    2. inputs:
        - result: 訓練摘要（需含 val_loss_history 或 label_counts）
    3. return:
        - bool: True=剔除
    """
    vh = result.get("val_loss_history") or (result.get("val_history", {}) or {}).get("loss")
    debug_prefix = "[collapse_mask]"
    if isinstance(vh, (list, tuple)) and len(vh) >= 2:
        try:
            vh_f = float(vh[0])
            vh_l = float(vh[-1])
            vh_best = min(float(x) for x in vh if x is not None)
        except Exception:
            vh_f = vh_l = vh_best = float("nan")
        if vh_l > vh_f:
            print(f"{debug_prefix} drop by val_loss trend: first={vh_f:.6f}, last={vh_l:.6f}, best={vh_best:.6f}")
            return True
        else:
            print(f"{debug_prefix} keep by val_loss trend: first={vh_f:.6f}, last={vh_l:.6f}, best={vh_best:.6f}")

    lc = (result.get("label_counts", {}) or {})
    counts = lc.get("val") or lc.get("train") or {}
    try:
        n0 = int(counts.get(0, counts.get("0", 0)))
        n1 = int(counts.get(1, counts.get("1", 0)))
        tot = n0 + n1
        if tot > 0:
            pos_ratio = n1 / float(tot)
            if pos_ratio > 0.8 or pos_ratio < 0.2:
                print(f"{debug_prefix} drop by pos_ratio={pos_ratio:.4f} (n0={n0}, n1={n1})")
                return True
            else:
                print(f"{debug_prefix} keep by pos_ratio={pos_ratio:.4f} (n0={n0}, n1={n1})")
    except Exception:
        pass

    return False

# ---------------- main export ----------------
def export_tbm_predictions_for_trial(
    *,
    cfg: Dict,
    df_index: pd.DatetimeIndex,
    folds: List[Dict],
    fold_models: List[Tuple[Any, Dict, Dict]],  # (model/checkpoint, fold_dict, result)
    trial_dir: str | Path | None = None,
    date_start: str,
    date_end: str,
    src_tbm_csv_path: str | None = None,
    save_to_path: str | None = None,
    output_column: str = "pred",
    threshold_override: float | None = None,
    decision_mode: str = "mean_prob",   # "mean_prob" | "fold_vote" | "both"
    collapse_mask_enable: bool = True,
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
       - fold_models: List[Tuple(model_ref, fold_dict, result)]  模型或 checkpoint 路徑與對應 fold 訓練結果（含 best_val_thresh/temperature）
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

    device = str(cfg.get("device", "cuda")).lower()
    use_cuda = device.startswith("cuda")
    trial_dir_path: Optional[Path] = Path(trial_dir) if trial_dir is not None else None

    # AMP 選擇
    dtype = None
    if bool(cfg["train"].get("amp", True)):
        kind = str(cfg["train"].get("amp_dtype", "auto")).lower()
        if kind in {"fp16", "float16"}:
            dtype = torch.float16
        elif kind in {"bf16", "bfloat16"}:
            dtype = torch.bfloat16
        else:
            if use_cuda:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # ---- features ----
    feat_df = load_precomputed_features(path=str(cfg["data"]["path"])).astype(np.float32)

    micro_cfg = (cfg.get("data", {}) or {}).get("micro", {})
    if micro_cfg.get("enabled") and micro_cfg.get("path"):
        micro_df = load_precomputed_features(path=str(micro_cfg["path"])).astype(np.float32)
        feat_df = feat_df.join(micro_df, how="left")
        if feat_df.isna().any().any():
            raise ValueError("[tbm_exporter] Joined micro features still contain NaN/Inf. Please sanitize upstream.")

    feat_cols_all = [c for c in feat_df.columns if np.issubdtype(feat_df[c].dtype, np.number)]
    feat_df = feat_df.loc[:, feat_cols_all].astype(np.float32, copy=False)
    idx_all = pd.DatetimeIndex(feat_df.index)  # UTC, sorted

    # ---- TBM (完整表，保留為輸出底稿) ----
    tbm_path = str(src_tbm_csv_path) if (src_tbm_csv_path not in (None, "", "None", "none", "NULL", "null")) else str(cfg["label"]["tbm_csv_path"])
    if not os.path.exists(tbm_path):
        raise FileNotFoundError(f"TBM label CSV not found: {tbm_path}")

    tbm_all = pd.read_csv(tbm_path, parse_dates=["t0"])
    if "__rid" not in tbm_all.columns:
        tbm_all["__rid"] = np.arange(len(tbm_all), dtype=np.int64)

    t0u_all = ensure_utc_index(tbm_all["t0"])

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

    ts_start = pd.Timestamp(date_start, tz="UTC")
    ts_end = pd.Timestamp(date_end, tz="UTC")
    mask_date = (t0u_all >= ts_start) & (t0u_all <= ts_end)

    tbm_sel = tbm_all.loc[mask_side & mask_date].copy()
    t0u_sel = t0u_all[mask_side & mask_date]
    if tbm_sel.empty:
        raise ValueError("指定期間內（且符合 side 遮罩）的 TBM 事件為空，無法推論匯出。")

    # ---- 對齊時間（僅為推論候選）----
    align_method = str(cfg.get("label", {}).get("align_method", "pad")).lower()
    allowed_align = align_times(t0u_sel, idx_all, align_method)

    # === 準備 rid 索引（靜態，不會被消耗） ===
    tbm_sel = tbm_sel.copy()
    tbm_sel["t0u"] = ensure_utc_index(tbm_sel["t0"])
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

    # --- 品質閘：用 collapse_mask 剔除壞掉的 fold ---
    _kept, _dropped = [], []
    for i, (m, fdict, res) in enumerate(fold_models):
        try:
            if collapse_mask_enable:
                if collapse_mask(res):
                    _dropped.append({"fold_idx": i})
                    continue
        except Exception as e:
            # 資料缺欄等異常：保守起見留著，並提示
            print(f"[export][warn] collapse_mask failed on fold {i}: {e}; keep it.")
        _kept.append((m, fdict, res))

    if not _kept:
        raise RuntimeError(
            "[export] 無可用 fold 供匯出（可能因 collapse_mask 剔除或 fold_models 為空）。"
        )
    if _dropped:
        print(f"[export] drop {len(_dropped)} fold(s) by collapse_mask: {_dropped}")

    fold_models = _kept

    for fold_idx, (model_ref, fold_d, result) in enumerate(fold_models):
        # 每個 fold 自己的 t0 配對指標（不互相影響）
        rid_counter = defaultdict(int)  # key: t0_utc -> 已分配次數（當作 rid_lists_by_t0 的索引）

        feat_cols_current = feat_cols_all

        result = result or {}
        checkpoint_path = _resolve_checkpoint_path(model_ref, trial_dir_path, fold_idx)
        if checkpoint_path is not None:
            model, feat_cols_current = _load_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                cfg=cfg,
                feat_df=feat_df,
                default_cols=feat_cols_current,
                result=result,
                fold_idx=fold_idx,
            )
        elif hasattr(model_ref, "state_dict"):
            model = model_ref
            ckpt_cols = result.get("_feature_columns")
            if ckpt_cols:
                feat_cols_current = [c for c in ckpt_cols if c in feat_df.columns]
        else:
            print(f"[tbm_exporter][warn] fold {fold_idx} has no usable model reference; skip.")
            continue

        # 1) split：決定 scaler 擬合窗口（train）
        df_idx = pd.DataFrame(index=pd.DatetimeIndex(df_index))
        tr_idx, va_idx, te_idx = split_fold_to_indices(df_idx, fold_d, cfg)
        train_align = align_times(tr_idx, idx_all, align_method)

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
        cols_to_scale = pick_cols_to_scale(feat_df.loc[fit_index, feat_cols_current], feat_cols_current)
        scaler_kind = cfg["sequence"]["scaler"]
        min_frac = float(cfg["sequence"].get("min_frac", 0.2))
        scaler = _get_scaler(scaler_kind, window=L, min_frac=min_frac)

        if hasattr(scaler, "is_timesafe") and getattr(scaler, "is_timesafe", False):
            feat_scaled = scaler.transform_full(feat_df, cols_to_scale=cols_to_scale)
            feat_scaled = feat_scaled.loc[:, feat_cols_current]
        else:
            if scaler is None:
                feat_scaled = feat_df.loc[:, feat_cols_current]
            else:
                sk = ColumnSubsetScaler(scaler, all_cols=feat_cols_current, cols_to_scale=cols_to_scale)
                sk.fit_df(feat_df.loc[fit_index, feat_cols_current])
                arr = feat_df.loc[:, feat_cols_current].values.astype(np.float32, copy=False)
                arr = sk.transform(arr)
                feat_scaled = feat_df.loc[:, feat_cols_current].copy()
                feat_scaled.loc[:, feat_cols_current] = arr

        # 3) 建推論 Dataset（僅含 allowed_align 事件）
        ds_inf = EventDataset(
            feat_scaled,
            tbm_path,
            seq_len=L,
            feature_cols=feat_cols_current,
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
        if not hasattr(model, "to") or not hasattr(model, "eval"):
            raise TypeError("[tbm_exporter] Only torch.nn.Module models are supported for export.")
        model = model.to(device)
        model.eval()
        with torch.no_grad(), amp.autocast(
            device_type=("cuda" if use_cuda else "cpu"),
            dtype=dtype,
            enabled=(dtype is not None and use_cuda),
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
                        key_t0 = ensure_utc_index(pd.Index([key_t0]))[0]

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
            if use_cuda and torch.cuda.is_available():
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
    mask_output = (t0u_all >= ts_start) & (t0u_all <= ts_end)
    out_df = tbm_all.loc[mask_output].copy()
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

        # ---- final metrics與混淆矩陣 ----
        metrics_info = None
        cm_info = None
        label_col = next((c for c in ("label", "tbm_label", "y", "target") if c in out_df.columns), None)
        if label_col is not None and mask_out.any():
            candidate_series: List[Tuple[str, pd.Series]] = []
            if dm in {"mean_prob", "both"} and col_prob_pred in out_df.columns:
                candidate_series.append(("soft_prob", out_df[col_prob_pred]))
            if dm in {"fold_vote", "both"} and col_vote_pred in out_df.columns:
                candidate_series.append(("hard_vote", out_df[col_vote_pred]))

            for mode, series in candidate_series:
                preds = series[mask_out].dropna()
                if preds.empty:
                    continue
                y_true_series = out_df.loc[preds.index, label_col].dropna()
                common_idx = preds.index.intersection(y_true_series.index)
                if len(common_idx) == 0:
                    continue
                y_pred_vals = preds.loc[common_idx].astype(int).to_numpy()
                y_true_vals = y_true_series.loc[common_idx].astype(int).to_numpy()
                classes = np.unique(y_true_vals)
                if classes.size == 0:
                    continue
                avg = "binary" if classes.size == 2 else "macro"
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_vals, y_pred_vals, average=avg, zero_division=0
                )
                accuracy = float(np.mean(y_pred_vals == y_true_vals))
                mcc_val = float(matthews_corrcoef(y_true_vals, y_pred_vals))
                cm = confusion_matrix(y_true_vals, y_pred_vals, labels=np.sort(classes))
                class_labels = [str(c) for c in np.sort(classes)]

                cm_plot_path = Path(os.path.splitext(save_to_path)[0] + f"_{mode}_confusion_matrix.png")
                cm_plot_path.parent.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks(range(len(class_labels)))
                ax.set_xticklabels(class_labels)
                ax.set_yticks(range(len(class_labels)))
                ax.set_yticklabels(class_labels)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title(f"Confusion Matrix ({mode})")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                max_val = cm.max() if cm.size else 0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(
                            j,
                            i,
                            int(cm[i, j]),
                            ha="center",
                            va="center",
                            color="white" if max_val and cm[i, j] > max_val / 2 else "black",
                        )
                plt.tight_layout()
                fig.savefig(cm_plot_path, dpi=200)
                plt.close(fig)

                metrics_info = {
                    "mode": mode,
                    "average": avg,
                    "support": int(len(y_true_vals)),
                    "accuracy": accuracy,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "mcc": mcc_val,
                }
                cm_info = {
                    "mode": mode,
                    "labels": class_labels,
                    "matrix": cm.astype(int).tolist(),
                    "plot_path": os.path.relpath(cm_plot_path, start=os.path.dirname(save_to_path)),
                }
                break

        if metrics_info is not None:
            summary["metrics"] = metrics_info
        if cm_info is not None:
            summary["confusion_matrix"] = cm_info

        with open(os.path.splitext(save_to_path)[0] + ".summary.json", "w", encoding="utf-8") as jf:
            json.dump(summary, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return save_to_path
