from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from trader.predictor.model import build_model
except ModuleNotFoundError:
    from predictor.model import build_model
import re

from typing import Literal
# ------------------------------------------------------------
# Predictor (inference-only)
# ------------------------------------------------------------

class Predictor:
    """
    1. 說明:
        Trader 端輕量推論器：載入單一 checkpoint，預先建模並在 predict 時直接前向。
    2. inputs:
        - model_path_list: checkpoint 路徑列表
        - device: 預設 "cuda:0"
    3. return:
        - Predictor instance
    """

    def __init__(self, cfg: Dict[str, Any], side: Literal["long", "short"]):
        if side == "long":
            model_path_list = cfg["model_path_list"]["long"]
        elif side == "short":
            model_path_list = cfg["model_path_list"]["short"]
        else: 
            raise ValueError("side should be long or short.")

        if not model_path_list:
            raise ValueError("cfg.model_path_list must contain at least one checkpoint.")
        seq_len_cfg = cfg.get("seq_len", None)
        if seq_len_cfg is None:
            raise ValueError("cfg.seq_len is required.")
        device_cfg = cfg.get("device", "cuda:0")

        self.device = device_cfg if torch.cuda.is_available() and str(device_cfg).startswith("cuda") else "cpu"
        self.seq_len = int(seq_len_cfg)
        vote_mode_cfg = str(cfg.get("vote_mode", "hard")).strip().lower()
        if vote_mode_cfg not in {"hard", "soft"}:
            raise ValueError("cfg.vote_mode must be 'hard' or 'soft'.")
        self.vote_mode = vote_mode_cfg

        self.models: List[nn.Module] = []
        self.model_meta: List[Dict[str, Any]] = []
        self.feature_columns: List[str] = []
        self.num_classes: int | None = None
        for i, mp in enumerate(model_path_list):
            ckpt_path = Path(mp)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")

            feature_columns = ckpt.get("feature_columns", None)
            if not feature_columns:
                raise ValueError(f"[ckpt {ckpt_path}] missing feature_columns; cannot align features.")
            temperature = ckpt.get("temperature", None)
            if temperature is None:
                raise ValueError(f"[ckpt {ckpt_path}] missing temperature; refusing to use default.")

            if i == 0:
                self.feature_columns = list(feature_columns)
            else:
                if list(feature_columns) != self.feature_columns:
                    raise ValueError(f"[ckpt {ckpt_path}] feature_columns mismatch across checkpoints.")

            state_dict = ckpt.get("state_dict", ckpt)
            model_cfg = ckpt.get("model_cfg", ckpt.get("model", {})) or {}
            model_cfg = self._maybe_infer_layers_from_state_dict(model_cfg, state_dict)
            num_classes = self._infer_num_classes(model_cfg, state_dict)
            thr = ckpt.get("best_val_thresh", None)
            if num_classes <= 2 and thr is None:
                raise ValueError(
                    f"[ckpt {ckpt_path}] binary checkpoint requires best_val_thresh; got None."
                )

            n_features = len(self.feature_columns)
            model = build_model(model_cfg, n_features, self.feature_columns)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            if self.num_classes is None:
                self.num_classes = int(num_classes)
            elif int(num_classes) != int(self.num_classes):
                raise ValueError(
                    f"[ckpt {ckpt_path}] num_classes mismatch across checkpoints: "
                    f"{num_classes} != {self.num_classes}"
                )
            self.models.append(model)
            self.model_meta.append(
                {
                    "temperature": float(temperature),
                    "threshold": None if thr is None else float(thr),
                    "num_classes": int(num_classes),
                }
            )

    def predict(self, feat_df: pd.DataFrame, model_idx: int = 0) -> Tuple[float, Union[bool, int]]:
        """
        1. 說明:
            針對單一時間窗特徵做推論。
            - 二分類：回傳 bool（使用 best_val_thresh）
            - 多分類：回傳 int 類別索引（argmax）
        2. inputs:
            - feat_df: 單窗特徵 DataFrame，index 可有任意時間，需含所有 feature_columns，行數=seq_len
            - model_idx: 使用哪個 checkpoint（預設第0個）
        3. return:
            - (elapsed_sec, pred)
        """
        self._validate_features(feat_df)
        x = feat_df[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        self._validate_length_and_time(feat_df)
        if not np.isfinite(x).all():
            raise ValueError("feat_df contains NaN/Inf; please sanitize input.")

        xb = torch.from_numpy(x).unsqueeze(0).to(self.device)

        if model_idx < 0 or model_idx >= len(self.models):
            raise IndexError(f"model_idx {model_idx} out of range for {len(self.models)} models.")
        model = self.models[model_idx]
        meta = self.model_meta[model_idx]

        start = time.perf_counter()
        pred: Union[bool, int]
        with torch.no_grad():
            logits = model(xb) / meta["temperature"]
            if int(meta["num_classes"]) <= 2:
                thr = meta["threshold"]
                if thr is None:
                    raise ValueError("binary inference requires threshold, but got None.")
                p1 = torch.sigmoid(logits.float().squeeze(-1))
                prob = float(p1.item())
                pred = bool(prob >= float(thr))
            elif logits.ndim == 1:
                pred = int(torch.argmax(logits).item())
            else:
                pred = int(torch.argmax(logits.float(), dim=1).item())
        inference_time = time.perf_counter() - start

        return (inference_time, pred)

    def predict_vote(
        self,
        feat_df: pd.DataFrame,
        vote_mode: Literal["hard", "soft"] | None = None,
    ) -> Tuple[float, Union[bool, int]]:
        """
        1. 說明:
            多 checkpoint 投票：
            - hard:
                - 二分類：各模型各自 threshold 判斷後做多數決
                - 多分類：各模型 argmax 後做多數決
            - soft:
                - 二分類：先平均各模型 p1，再用 threshold（平均 fold threshold）判斷
                - 多分類：先平均各模型 class probabilities，再 argmax
        2. inputs:
            - feat_df: 單窗特徵 DataFrame，需符合長度/欄位/時間檢查
            - vote_mode: 'hard' 或 'soft'；None 時使用 cfg.vote_mode
        3. return:
            - (elapsed_sec, pred)
        """
        if not self.models:
            raise ValueError("No models loaded for predict_vote.")
        mode = str(vote_mode or self.vote_mode).strip().lower()
        if mode not in {"hard", "soft"}:
            raise ValueError("vote_mode must be 'hard' or 'soft'.")
        self._validate_features(feat_df)
        self._validate_length_and_time(feat_df)
        x = feat_df[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(x).all():
            raise ValueError("feat_df contains NaN/Inf; please sanitize input.")

        xb = torch.from_numpy(x).unsqueeze(0).to(self.device)

        is_binary = all(int(meta["num_classes"]) <= 2 for meta in self.model_meta)
        preds_bin: List[bool] = []
        probs_bin: List[float] = []
        preds_cls: List[int] = []
        probs_cls: List[np.ndarray] = []
        start = time.perf_counter()
        with torch.no_grad():
            for model, meta in zip(self.models, self.model_meta):
                logits = model(xb) / meta["temperature"]
                if int(meta["num_classes"]) <= 2:
                    thr = meta["threshold"]
                    if thr is None:
                        raise ValueError("binary ensemble inference requires threshold, but got None.")
                    p1 = torch.sigmoid(logits.float().squeeze(-1))
                    prob = float(p1.item())
                    probs_bin.append(prob)
                    preds_bin.append(bool(prob >= float(thr)))
                else:
                    if logits.ndim == 1:
                        logits2 = logits.float().unsqueeze(0)
                    else:
                        logits2 = logits.float()
                    p = torch.softmax(logits2, dim=1).squeeze(0).detach().cpu().numpy()
                    probs_cls.append(p.astype(np.float64, copy=False))
                    preds_cls.append(int(np.argmax(p)))
        inference_time = time.perf_counter() - start

        if is_binary:
            if mode == "soft":
                mean_prob = float(np.mean(np.asarray(probs_bin, dtype=np.float64)))
                thr_arr = np.asarray([float(meta["threshold"]) for meta in self.model_meta], dtype=np.float64)
                mean_thr = float(np.mean(thr_arr))
                pred: Union[bool, int] = bool(mean_prob >= mean_thr)
                return (inference_time, pred)

            votes_true = sum(1 for p in preds_bin if p)
            votes_false = len(preds_bin) - votes_true
            pred = votes_true > votes_false
            return (inference_time, pred)

        if mode == "soft":
            prob_mat = np.asarray(probs_cls, dtype=np.float64)
            mean_probs = prob_mat.mean(axis=0)
            pred = int(np.argmax(mean_probs))
            return (inference_time, pred)

        # hard 多分類：同票時採「最小 class id」作 deterministic tie-break。
        cls_values, counts = np.unique(np.asarray(preds_cls, dtype=np.int64), return_counts=True)
        winner_idx = int(np.argmax(counts))
        pred = int(cls_values[winner_idx])
        return (inference_time, pred)

    def _validate_features(self, df: pd.DataFrame) -> None:
        """
        1. 說明:
            檢查輸入特徵必備欄位與型態。
        2. inputs:
            - df: 特徵 DataFrame
        3. return:
            - None
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("feat_df must be a pandas DataFrame.")
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"feat_df missing required feature columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("feat_df.index must be a DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            raise ValueError("feat_df index must be monotonic increasing.")

    def _validate_length_and_time(self, df: pd.DataFrame) -> None:
        """
        1. 說明:
            確保序列長度與 15min 等距時間網格。
        2. inputs:
            - df: 特徵 DataFrame
        3. return:
            - None
        """
        if len(df.index) != self.seq_len:
            raise ValueError(f"feat_df length {len(df.index)} != required seq_len {self.seq_len}")
        # 15min 等距檢查
        diffs = df.index.to_series().diff().dropna().unique()
        if len(diffs) != 1 or diffs[0] != pd.Timedelta(minutes=15):
            raise ValueError("feat_df index must be contiguous with 15min frequency.")

    @staticmethod
    def _maybe_infer_layers_from_state_dict(model_cfg: Dict[str, Any], state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. 說明:
            從 state_dict 推斷 Transformer block 數，填入 model_cfg.n_layers/num_layers，避免層數不符。
        2. inputs:
            - model_cfg: 原始模型設定
            - state_dict: checkpoint 權重
        3. return:
            - model_cfg（可能更新 n_layers/num_layers）
        """
        max_idx = -1
        pattern_backbone = re.compile(r"backbone\.blocks\.(\d+)\.")
        pattern_blocks = re.compile(r"^blocks\.(\d+)\.")
        for k in state_dict.keys():
            m = pattern_backbone.search(k)
            if not m:
                m = pattern_blocks.search(k)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        if max_idx >= 0:
            n_layers = max_idx + 1
            model_cfg = dict(model_cfg)
            model_cfg["n_layers"] = n_layers
            model_cfg["num_layers"] = n_layers
        return model_cfg

    @staticmethod
    def _infer_num_classes(model_cfg: Dict[str, Any], state_dict: Dict[str, Any]) -> int:
        """
        Infer number of classes from model config/state dict.

        Args:
            model_cfg: Model config loaded from checkpoint.
            state_dict: Model state dict loaded from checkpoint.
        Returns:
            Number of classes. Falls back to 1 when unavailable.
        """
        cfg_n = model_cfg.get("num_classes", None)
        if cfg_n is not None:
            try:
                n = int(cfg_n)
                if n >= 1:
                    return n
            except Exception:
                pass
        for k, v in state_dict.items():
            if str(k).endswith("head.weight") and hasattr(v, "shape") and len(v.shape) >= 1:
                try:
                    n = int(v.shape[0])
                    if n >= 1:
                        return n
                except Exception:
                    continue
        return 1
