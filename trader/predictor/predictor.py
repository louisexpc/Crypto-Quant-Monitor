from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import amp, nn

from trader.predictor.model import build_model
import re


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

    def __init__(self, cfg: Dict[str, Any], side: str):
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

        self.models: List[nn.Module] = []
        self.model_meta: List[Dict[str, Any]] = []
        self.feature_columns: List[str] = []

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
            thr = ckpt.get("best_val_thresh", None)
            if thr is None:
                raise ValueError(f"[ckpt {ckpt_path}] missing best_val_thresh; refusing to use default.")

            amp_cfg = ckpt.get("amp", None)
            amp_dtype_cfg = ckpt.get("amp_dtype", None)
            if amp_cfg is None or amp_dtype_cfg is None:
                raise ValueError(f"[ckpt {ckpt_path}] missing amp/amp_dtype; refusing to use default.")
            amp_dtype_cfg = str(amp_dtype_cfg).lower()

            if i == 0:
                self.feature_columns = list(feature_columns)
                self.amp_enabled = bool(amp_cfg)
                if amp_dtype_cfg in {"bf16", "bfloat16"}:
                    self.amp_dtype = torch.bfloat16
                elif amp_dtype_cfg in {"fp16", "float16"}:
                    self.amp_dtype = torch.float16
                else:
                    raise ValueError(f"[ckpt {ckpt_path}] unsupported amp_dtype: {amp_dtype_cfg}")
            else:
                if list(feature_columns) != self.feature_columns:
                    raise ValueError(f"[ckpt {ckpt_path}] feature_columns mismatch across checkpoints.")
                # 若 amp 設定不同，沿用第一個 ckpt 的設定並提醒
                if bool(amp_cfg) != self.amp_enabled or str(amp_dtype_cfg) != str(self.amp_dtype).lower():
                    print(f"[Predictor][WARN] amp config mismatch in {ckpt_path}; using first ckpt's amp settings.")

            state_dict = ckpt.get("state_dict", ckpt)
            model_cfg = ckpt.get("model_cfg", ckpt.get("model", {})) or {}
            model_cfg = self._maybe_infer_layers_from_state_dict(model_cfg, state_dict)

            n_features = len(self.feature_columns)
            model = build_model(model_cfg, n_features, self.feature_columns)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            self.models.append(model)
            self.model_meta.append(
                {
                    "temperature": float(temperature),
                    "threshold": float(thr),
                }
            )

    def predict(self, feat_df: pd.DataFrame, model_idx: int = 0) -> Tuple[float, bool]:
        """
        1. 說明:
            針對單一時間窗特徵做推論，回傳 (推論耗時秒, 預測標籤bool)。
        2. inputs:
            - feat_df: 單窗特徵 DataFrame，index 可有任意時間，需含所有 feature_columns，行數=seq_len
            - model_idx: 使用哪個 checkpoint（預設第0個）
        3. return:
            - (elapsed_sec, pred_bool)
        """
        self._validate_features(feat_df)
        x = feat_df[self.feature_columns].to_numpy(dtype=np.float32, copy=False)
        self._validate_length_and_time(feat_df)
        if not np.isfinite(x).all():
            raise ValueError("feat_df contains NaN/Inf; please sanitize input.")

        xb = torch.from_numpy(x).unsqueeze(0).to(self.device)

        if model_idx < 0 or model_idx >= len(self.models):
            raise IndexError(f"model_idx {model_idx} out of range for {len(self.models)} models.")
        model = self.models[model_idx]
        meta = self.model_meta[model_idx]

        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        dtype = self.amp_dtype if self.amp_enabled else None

        start = time.perf_counter()
        probs: List[float] = []
        with torch.no_grad(), amp.autocast(device_type=device_type, dtype=dtype, enabled=self.amp_enabled):
            logits = model(xb) / meta["temperature"]
            if logits.ndim == 1 or logits.shape[-1] == 1:
                p1 = torch.sigmoid(logits.float().squeeze(-1))
            else:
                p1 = torch.softmax(logits.float(), dim=1)[:, 1]
            probs.append(float(p1.item()))
        inference_time = time.perf_counter() - start

        prob = float(np.mean(probs))
        pred_bool = bool(prob >= meta["threshold"])
        return (inference_time, pred_bool)

    def predict_vote(self, feat_df: pd.DataFrame) -> Tuple[float, bool]:
        """
        1. 說明:
            多 checkpoint 投票：逐一模型推論，平均耗時，並依個別門檻計算票數。
        2. inputs:
            - feat_df: 單窗特徵 DataFrame，需符合長度/欄位/時間檢查
        3. return:
            - (elapsed_sec, pred_bool)
        """
        if not self.models:
            raise ValueError("No models loaded for predict_vote.")
        self._validate_features(feat_df)
        self._validate_length_and_time(feat_df)
        x = feat_df[self.feature_columns].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(x).all():
            raise ValueError("feat_df contains NaN/Inf; please sanitize input.")

        xb = torch.from_numpy(x).unsqueeze(0).to(self.device)
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        dtype = self.amp_dtype if self.amp_enabled else None

        preds: List[bool] = []
        start = time.perf_counter()
        with torch.no_grad(), amp.autocast(device_type=device_type, dtype=dtype, enabled=self.amp_enabled):
            for model, meta in zip(self.models, self.model_meta):
                logits = model(xb) / meta["temperature"]
                if logits.ndim == 1 or logits.shape[-1] == 1:
                    p1 = torch.sigmoid(logits.float().squeeze(-1))
                else:
                    p1 = torch.softmax(logits.float(), dim=1)[:, 1]
                prob = float(p1.item())
                preds.append(bool(prob >= meta["threshold"]))
        inference_time = time.perf_counter() - start

        votes_true = sum(1 for p in preds if p)
        votes_false = len(preds) - votes_true
        pred_bool = votes_true > votes_false
        return (inference_time, pred_bool)

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
