# train/inference/predictor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import amp

from train.data.dataset.event_dataset import EventDataset
from train.data.dataloaders.base import align_times, ensure_utc_index
from train.models.model_factory import build_model


PathLike = Union[str, Path]


@dataclass
class _InferPack:
    """
    1. 說明:
        封裝「一次推論」需要的資料結構，讓多個模型可以共用同一份 dataset/dataloader。
    2. inputs:
        - tbm_out: 輸出底稿（date window 內的 TBM events；未必都會被預測）
        - dl: 推論 dataloader（只包含 keep_sides + date window 內的事件）
        - rid_by_sample: 每個 sample 對應的 __rid（長度等於 dataset events）
        - feature_cols: 實際用來推論的欄位（與 checkpoint 對齊後）
        - date_start/date_end: 這次推論的區間（UTC）
    3. return:
        - _InferPack 物件
    """
    tbm_out: pd.DataFrame
    dl: DataLoader
    rid_by_sample: List[int]
    feature_cols: List[str]
    date_start: pd.Timestamp
    date_end: pd.Timestamp


class Predictor:
    """
    1. 說明:
        TBM 推論器：給定 precomputed feature df 與 checkpoint path，產出預測 df。
        - predict(): 單一模型預測
        - predict_vote(): 多模型（多 fold checkpoint）投票
    2. inputs:
        - cfg: 全域設定 dict（device/seq_len/keep_sides/tbm_csv_path/date window/amp 等）
    3. return:
        - Predictor instance
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        1. 說明:
            初始化 Predictor（只保存 cfg 與裝置/AMP 設定）。
        2. inputs:
            - cfg: 設定 dict（建議與 train 同一份）
        3. return:
            - None
        """
        self.cfg = cfg
        self.device = str(cfg.get("device", "cpu"))
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"

    # ----------------------------
    # public API
    # ----------------------------
    def predict(
        self,
        df: pd.DataFrame,
        tbm_df: pd.DataFrame,
        model_path: PathLike,
        date_start: Optional[pd.Timestamp] = None,
        date_end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        1. 說明:
            單一 checkpoint 推論：回傳「date window 內」的 TBM events df，並填上 pred 欄位。
            - 事件來源：呼叫端提供的 tbm_df（需含 t0/side/label，__rid 若缺會自動補）
            - 推論區間：date_start/date_end（若未提供，取事件/特徵可用時間交集）
        2. inputs:
            - df: precomputed features（時間 index，numeric columns）
            - tbm_df: TBM 事件 DataFrame
            - model_path: checkpoint 檔案路徑（.pt/.pth/.ckpt）
            - date_start/date_end: 推論時間窗（可選）
        3. return:
            - out_df: 含 __rid 與單模型預測欄位的 DataFrame
        """
        pack = self._build_infer_pack(df, tbm_df, date_start=date_start, date_end=date_end)
        pred_df = self._predict_on_pack(pack, Path(model_path), suffix=None)
        return pred_df

    def predict_vote(
        self,
        df: pd.DataFrame,
        tbm_df: pd.DataFrame,
        model_paths_or_dir: Union[PathLike, Sequence[PathLike]],
        date_start: Optional[pd.Timestamp] = None,
        date_end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        1. 說明:
            多模型投票：會跑多輪推論（模型數 = 找到的 checkpoint 數），回傳帶 vote 欄位的 df。
        2. inputs:
            - df: precomputed features（時間 index，numeric columns）
            - tbm_df: TBM 事件 DataFrame
            - model_paths_or_dir:
                - (a) checkpoint 檔案路徑列表
                - (b) 目錄：自動遞迴搜尋常見 best checkpoint 檔
            - date_start/date_end: 推論時間窗（可選）
        3. return:
            - out_df: 含 pred_i + pred_vote 的 DataFrame
        """
        model_paths = self._normalize_model_paths(model_paths_or_dir)
        if len(model_paths) == 0:
            raise ValueError("[Predictor] No checkpoint found for predict_vote().")

        pack = self._build_infer_pack(df, tbm_df, date_start=date_start, date_end=date_end)
        prefix = "pred"

        out_df = pack.tbm_out.copy()
        per_model_cols: List[str] = []

        for i, mp in enumerate(model_paths):
            col_i = f"{prefix}_{i}"
            df_i = self._predict_on_pack(pack, mp, suffix=str(i))
            # 只 merge 這次模型的欄位，避免覆蓋其他模型結果
            cols_to_keep = [c for c in ["__rid", col_i, f"{col_i}_p1"] if c in df_i.columns]
            out_df = out_df.merge(df_i[cols_to_keep], on="__rid", how="left", validate="one_to_one", suffixes=("", "_dup"))
            # 清掉 merge 產生的重複欄
            dup_cols = [c for c in out_df.columns if c.endswith("_dup")]
            if dup_cols:
                out_df = out_df.drop(columns=dup_cols)
            if col_i in out_df.columns:
                per_model_cols.append(col_i)

        # vote 聚合（沿用你原本 exporter 的欄位命名）
        col_vote_total = f"{prefix}_vote_votes_total"
        col_vote_margin = f"{prefix}_vote_margin"
        col_vote_pred = f"{prefix}_vote"

        pred_mat = out_df[per_model_cols]
        votes_total = pred_mat.notna().sum(axis=1).astype("Int64")
        votes_one = pred_mat.fillna(0).astype(float).sum(axis=1)
        votes_zero = votes_total.astype(float) - votes_one
        margin = votes_one - votes_zero

        out_df[col_vote_total] = votes_total
        out_df[col_vote_margin] = margin
        out_df[col_vote_pred] = (votes_one > votes_zero).astype("Int64")

        # 沒有任何模型投票的 row（例如非 keep_sides）保持 NA
        no_vote_mask = votes_total.isna() | (votes_total == 0)
        out_df.loc[no_vote_mask, [col_vote_margin, col_vote_pred]] = pd.NA

        return out_df

    # ----------------------------
    # core building blocks
    # ----------------------------
    def _build_infer_pack(
        self,
        feat_df: pd.DataFrame,
        tbm_df: pd.DataFrame,
        date_start: Optional[pd.Timestamp] = None,
        date_end: Optional[pd.Timestamp] = None,
    ) -> _InferPack:
        """
        1. 說明:
            用 cfg 的 TBM CSV + date window + keep_sides，把推論 dataset/dataloader 建好（一次就好）。
            這份 pack 可以被多個 checkpoint 共用，避免每個模型都重建 dataset。
        2. inputs:
            - feat_df: precomputed features df（時間 index）
            - tbm_df: TBM events df（需含 t0/side/label；__rid 若缺會自動補）
            - date_start/date_end: 推論窗口，若 None 會取 tbm/feat 的時間交集
        3. return:
            - _InferPack
        """
        feat_df = self._sanitize_feat_df(feat_df)

        tbm_all = tbm_df.copy()
        if "t0" not in tbm_all.columns:
            raise ValueError("[Predictor] tbm_df must contain column 't0'.")
        if "side" not in tbm_all.columns:
            raise ValueError("[Predictor] tbm_df must contain column 'side'.")
        if "label" not in tbm_all.columns:
            raise ValueError("[Predictor] tbm_df must contain column 'label'.")

        if "__rid" not in tbm_all.columns:
            tbm_all["__rid"] = np.arange(len(tbm_all), dtype=np.int64)

        t0u_all = ensure_utc_index(tbm_all["t0"])
        feat_idx_all = ensure_utc_index(feat_df.index)
        if len(t0u_all) == 0:
            raise ValueError("[Predictor] tbm_df is empty.")

        ds = self._to_utc_ts(date_start) if date_start is not None else None
        de = self._to_utc_ts(date_end) if date_end is not None else None
        if ds is None:
            ds = max(t0u_all.min(), feat_idx_all.min())
        if de is None:
            de = min(t0u_all.max(), feat_idx_all.max())

        mask_date = (t0u_all >= ds) & (t0u_all <= de)

        # 輸出底稿：date window 內的事件（即使 non-keep_sides 也保留，方便對齊你原本行為）
        tbm_out = tbm_all.loc[mask_date].copy()

        # keep_sides 過濾（只會對這些事件做推論）
        keep_sides = str(self.cfg.get("label", {}).get("keep_sides", "both")).lower()
        side_i = tbm_all["side"].map(self._side_to_int)

        if keep_sides == "long":
            mask_side = side_i.eq(1)
        elif keep_sides == "short":
            mask_side = side_i.eq(-1)
        else:
            mask_side = side_i.isin([1, -1])

        tbm_sel = tbm_all.loc[mask_date & mask_side].copy()
        t0u_sel = t0u_all[mask_date & mask_side]
        if tbm_sel.empty:
            raise ValueError("[Predictor] TBM events empty after (date window + keep_sides).")

        # 對齊時間
        idx_all = pd.DatetimeIndex(feat_df.index)  # UTC, sorted
        align_method = str(self.cfg.get("label", {}).get("align_method", "pad")).lower()
        allowed_align = align_times(t0u_sel, idx_all, align_method)

        # feature columns（這裡先用 numeric 全欄；真正與 checkpoint 對齊會在 _predict_on_pack 裡做）
        feat_cols_numeric = [c for c in feat_df.columns if np.issubdtype(feat_df[c].dtype, np.number)]
        feat_view = feat_df.loc[:, feat_cols_numeric]

        # 建 dataset / loader
        L = int(self.cfg.get("sequence", {}).get("seq_len", 144))
        bs = int(((self.cfg.get("post_infer", {}) or {}).get("tbm_concat", {}) or {}).get("batch_size",
                 (self.cfg.get("train", {}) or {}).get("batch_size", 256)))

        ds = EventDataset(
            feat_view,
            tbm_df=tbm_all,
            seq_len=L,
            feature_cols=feat_cols_numeric,
            keep_sides=keep_sides,
            align_method=align_method,
            device="cpu",
            allowed_align_index=allowed_align,
        )
        dl = DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )

        # 建立 t0 -> rid_list（穩定對應）
        tbm_sel["t0u"] = ensure_utc_index(tbm_sel["t0"])
        events_order = tbm_sel.sort_values(["t0u", "__rid"])
        rid_lists_by_t0: Dict[pd.Timestamp, List[int]] = {}
        for rid, t0u in zip(events_order["__rid"].to_numpy(), events_order["t0u"].to_numpy()):
            rid_lists_by_t0.setdefault(pd.Timestamp(t0u), []).append(int(rid))

        # precompute rid_by_sample（避免 forward loop 裡一直摸 ds.events）
        rid_by_sample: List[int] = []
        rid_counter = defaultdict(int)
        for ev in ds.events:
            key_t0 = getattr(ev, "t0_utc", None)
            if key_t0 is None:
                key_t0 = getattr(ev, "t0", getattr(ev, "time", None))
                key_t0 = ensure_utc_index(pd.Index([key_t0]))[0]

            i_assign = rid_counter[key_t0]
            rid_list = rid_lists_by_t0.get(key_t0, [])
            if i_assign >= len(rid_list):
                rid_by_sample.append(-1)
            else:
                rid_by_sample.append(rid_list[i_assign])
                rid_counter[key_t0] += 1

        return _InferPack(
            tbm_out=tbm_out,
            dl=dl,
            rid_by_sample=rid_by_sample,
            feature_cols=feat_cols_numeric,
            date_start=ds,
            date_end=de,
        )

    def _predict_on_pack(self, pack: _InferPack, model_path: Path, suffix: Optional[str] = None) -> pd.DataFrame:
        """
        1. 說明:
            對同一份 pack（dataset/dataloader）跑單一 checkpoint，並把結果 merge 回 tbm_out。
        2. inputs:
            - pack: _InferPack（共用資料）
            - model_path: checkpoint path
            - out_col: 基底欄位名（例如 "pred"）
            - suffix: 若不為 None，寫入欄位為 f"{out_col}_{suffix}"；否則寫入 f"{out_col}_0" 不做
        3. return:
            - out_df: pack.tbm_out + 預測欄位
        """
        prefix = "pred"
        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        ckpt_cols = ckpt.get("feature_columns", None)
        ckpt_model_cfg = ckpt.get("model_cfg", None)
        temperature = float(ckpt.get("temperature", 1.0) or 1.0)
        thr = ckpt.get("best_val_thresh", None)
        thr_used = float(thr) if thr is not None else 0.5

        # feature cols 對齊（以 checkpoint 為主）
        feat_cols = pack.feature_cols
        if isinstance(ckpt_cols, (list, tuple)) and len(ckpt_cols) > 0:
            feat_cols = [c for c in ckpt_cols if c in feat_cols]
            if len(feat_cols) == 0:
                raise ValueError(f"[Predictor] checkpoint feature_columns not found in feat_df: {model_path}")

        cfg_local = dict(self.cfg)
        if ckpt_model_cfg is not None:
            cfg_local = dict(cfg_local)
            cfg_local["model"] = ckpt_model_cfg

        model = build_model(cfg_local, len(feat_cols), feat_cols)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        dtype, autocast_enabled = self._resolve_amp()
        device_type = self.device_type

        # forward
        pred_records: List[Tuple[int, float, int]] = []
        ptr = 0
        with torch.no_grad(), amp.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
            for Xb, _ in pack.dl:
                Xb = Xb.to(self.device, non_blocking=False)
                logits = model(Xb) / temperature

                if logits.ndim == 1 or logits.shape[-1] == 1:
                    p1 = torch.sigmoid(logits.float().squeeze(-1))
                else:
                    p1 = torch.softmax(logits.float(), dim=1)[:, 1]

                p1_np = p1.detach().to("cpu").numpy().astype(np.float32)
                rids = pack.rid_by_sample[ptr: ptr + len(p1_np)]

                for rid, prob in zip(rids, p1_np):
                    if rid < 0:
                        continue
                    yhat = int(float(prob) >= thr_used)
                    pred_records.append((rid, float(prob), yhat))

                ptr += len(p1_np)

        model = model.to("cpu")
        if self.device_type == "cuda":
            torch.cuda.empty_cache()

        # merge 回輸出底稿
        out_df = pack.tbm_out.copy()
        col_name = f"{prefix}_{suffix}" if suffix is not None else prefix
        proba_col = f"{col_name}_p1"

        out_df[col_name] = pd.NA
        out_df[proba_col] = pd.NA

        if len(pred_records) > 0:
            tmp = pd.DataFrame(pred_records, columns=["__rid", proba_col, col_name])
            out_df = out_df.merge(tmp, on="__rid", how="left", validate="one_to_one", suffixes=("", "_new"))
            if f"{col_name}_new" in out_df.columns:
                out_df[col_name] = out_df[f"{col_name}_new"]
                out_df = out_df.drop(columns=[f"{col_name}_new"])
            if f"{proba_col}_new" in out_df.columns:
                out_df[proba_col] = out_df[f"{proba_col}_new"]
                out_df = out_df.drop(columns=[f"{proba_col}_new"])

        return out_df
    # ----------------------------
    # misc helpers
    # ----------------------------
    def _sanitize_feat_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. 說明:
            確保 feature df 的 index 為 UTC DatetimeIndex、排序、且數值欄位可用。
        2. inputs:
            - df: feature df
        3. return:
            - df_sanitized
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("[Predictor] feat_df.index must be a DatetimeIndex.")

        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")

        df = df.copy()
        df.index = pd.DatetimeIndex(idx)
        df = df.sort_index()

        # numeric cast + NaN check
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(num_cols) == 0:
            raise ValueError("[Predictor] feat_df has no numeric columns.")
        if df[num_cols].isna().any().any():
            raise ValueError("[Predictor] feat_df contains NaN in numeric columns; please sanitize upstream.")

        return df

    @staticmethod
    def _to_utc_ts(ts: Any) -> pd.Timestamp:
        """
        1. 說明:
            將任意時間轉為 UTC tz-aware Timestamp。
        2. inputs:
            - ts: 可轉為 pandas Timestamp 的物件
        3. return:
            - pd.Timestamp (tz-aware, UTC)
        """
        ts_obj = pd.Timestamp(ts)
        if ts_obj.tzinfo is None:
            return ts_obj.tz_localize("UTC")
        return ts_obj.tz_convert("UTC")

    def _resolve_amp(self) -> Tuple[Optional[torch.dtype], bool]:
        """
        1. 說明:
            根據 cfg.train.amp / cfg.train.amp_dtype 決定 autocast dtype。
        2. inputs:
            - None
        3. return:
            - (dtype, enabled)
        """
        train = self.cfg.get("train", {}) or {}
        enabled = bool(train.get("amp", True))
        if not enabled:
            return None, False

        kind = str(train.get("amp_dtype", "auto")).lower()
        if kind in {"bf16", "bfloat16"}:
            return torch.bfloat16, True
        if kind in {"fp16", "float16"}:
            return torch.float16, True

        # auto
        if self.device_type == "cuda":
            return (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16), True
        return torch.bfloat16, True

    def _normalize_model_paths(self, x: Union[PathLike, Sequence[PathLike]]) -> List[Path]:
        """
        1. 說明:
            支援兩種輸入：
            - list of checkpoint paths
            - directory：自動找 best checkpoint
        2. inputs:
            - x: 路徑或路徑列表
        3. return:
            - checkpoints: List[Path]
        """
        if isinstance(x, (str, Path)):
            p = Path(x)
            if p.is_dir():
                pats = [
                    "**/checkpoints/*best*.ckpt",
                    "**/checkpoints/*best*.pt",
                    "**/checkpoints/*best*.pth",
                    "**/*best*.ckpt",
                    "**/*best*.pt",
                    "**/*best*.pth",
                ]
                out: List[Path] = []
                for pat in pats:
                    out.extend(list(p.glob(pat)))
                # 去重 + sort
                out = sorted({q.resolve() for q in out})
                return out
            return [p.resolve()]

        out2 = [Path(p).resolve() for p in x]
        return out2

    @staticmethod
    def _side_to_int(x: Any) -> Any:
        """
        1. 說明:
            side 欄位轉成 int：long=1, short=-1
        2. inputs:
            - x: side 原始值
        3. return:
            - int 或 np.nan
        """
        s = str(x).strip().lower()
        if s in {"long", "buy", "l", "+1"}:
            return 1
        if s in {"short", "sell", "s", "-1"}:
            return -1
        try:
            return int(float(s))
        except Exception:
            return np.nan
