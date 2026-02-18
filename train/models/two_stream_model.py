# train/models/two_stream_model.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple, List
import re
import torch
import torch.nn as nn

# 優先用新的 transformer_model；若專案仍保留舊檔名 Transformer.py，則回退兼容
try:
    from train.models.transformer_model import TemporalTransformer
except Exception:  # pragma: no cover
    from .transformer_model import TemporalTransformer  # 兼容舊路徑

from train.models.registry import register


def build_feature_indices_by_prefix(
    columns: Optional[Sequence[str]],
    minute_prefixes: Tuple[str, ...] = ("m_",),
) -> Tuple[List[int], List[int]]:
    """
    1. 說明: 依欄名 prefix 切出 minute 與 base 欄位的索引位置。
    2. inputs:
       - columns: 欄位名稱序列；None 代表無法切分（回傳空索引）
       - minute_prefixes: minute 前綴（可多個）
    3. return:
       - (minute_idx, base_idx): 兩個整數索引 list
    """
    if not columns:  # None 或空
        return [], []
    if isinstance(minute_prefixes, str):
        minute_prefixes = (minute_prefixes,)
    minute_idx = [
        i for i, c in enumerate(columns)
        if str(c).startswith(minute_prefixes) or re.match(r"^m\d+_", str(c))
    ]
    minute_idx_set = set(minute_idx)
    base_idx = [i for i, _ in enumerate(columns) if i not in minute_idx_set]
    return minute_idx, base_idx


class LSTMBlock(nn.Module):
    """
    可重用的 LSTM 子模組（序列 → 向量）。單向，取最後一步作輸出。
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        """
        1. 說明: 建立單向 LSTM encoder
        2. inputs:
           - input_size: 每時間步輸入維度
           - hidden_size: LSTM 隱層維度
           - num_layers: LSTM 層數
           - dropout: 層間 dropout（>1 層才生效）
        3. return: None
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.out_dim = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. 說明: 將 [B,S,F] 編碼為 [B,H]（取最後一步）
        2. inputs:
           - x: [B, S, F]
        3. return:
           - torch.Tensor: [B, H]
        """
        y, _ = self.lstm(x)     # [B, S, H]
        return y[:, -1, :]      # [B, H]


class TwoStreamHybrid(nn.Module):
    """
    輸入 x: [B, T, F]
    - 以欄名 prefix 切分 minute(1m 攤平) 與 base(15m)
    - minute ⇒ [B*T, S, Fm] ⇒ 單向 LSTM ⇒ [B*T, H] ⇒ [B,T,H]
    - 與 base 拼接 ⇒ 丟進 TemporalTransformer
    """
    def __init__(
        self,
        # minute stream
        minute_steps: int = 15,
        minute_hidden: int = 64,
        minute_layers: int = 1,
        minute_dropout: float = 0.0,

        # backbone (Transformer)
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        pooling: str = "attn",
        use_causal: bool = True,
        num_classes: int = 1,

        # 以欄名切分所需參數
        columns: Optional[Sequence[str]] = None,
        minute_prefixes: Tuple[str, ...] = ("m_",),
    ):
        """
        1. 說明: 建立雙流融合模型（1m LSTM + 15m Transformer）
        2. inputs:
           - minute_*: 分鐘流的 LSTM 超參
           - d_model/n_heads/...: Transformer 超參
           - pooling/use_causal/num_classes: Head 與遮罩行為
           - columns: 特徵欄位名（依 prefix 切 minute/base）
           - minute_prefixes: minute 前綴集合
        3. return: None
        """
        super().__init__()

        # ---- 1) 建 minute/base 索引 ----
        minute_idx, base_idx = build_feature_indices_by_prefix(columns, minute_prefixes)
        self.base_dim = len(base_idx)

        # ---- 2) 推斷 minute_feat ----
        if len(minute_idx) > 0:
            if len(minute_idx) % minute_steps != 0:
                raise ValueError(
                    f"len(minute_idx)={len(minute_idx)} 不能被 minute_steps={minute_steps} 整除；無法推斷 minute_feat。"
                )
            minute_feat = len(minute_idx) // minute_steps
        else:
            minute_feat = 0  # 無 minute 特徵

        self.minute_steps = int(minute_steps)
        self.minute_feat  = int(minute_feat)

        # 註冊索引為 buffer（隨 .to(device) 移動）
        self.register_buffer("minute_idx_buf", torch.as_tensor(minute_idx, dtype=torch.long), persistent=False)
        self.register_buffer("base_idx_buf",   torch.as_tensor(base_idx,   dtype=torch.long), persistent=False)

        # ---- 3) minute encoder（可選） ----
        if self.minute_feat > 0:
            self.min_block = LSTMBlock(
                input_size=self.minute_feat,
                hidden_size=minute_hidden,
                num_layers=minute_layers,
                dropout=minute_dropout,
            )
            self.min_out_dim = self.min_block.out_dim
        else:
            self.min_block = None
            self.min_out_dim = 0

        # ---- 4) Backbone ----
        in_dim_backbone = self.base_dim + self.min_out_dim
        if in_dim_backbone == 0:
            raise ValueError(
                "in_dim_backbone = 0：看起來既沒有 base 也沒有 minute 特徵。請檢查 columns/prefix 或特徵計畫。"
            )

        self.backbone = TemporalTransformer(
            input_dim    = in_dim_backbone,
            num_classes  = num_classes,
            d_model      = d_model,
            n_heads      = n_heads,
            num_layers   = num_layers,
            mlp_ratio    = mlp_ratio,
            dropout      = dropout,
            attn_dropout = attn_dropout,
            pooling      = pooling,
            use_causal   = use_causal,
        )
        self._shape_debug_printed = False

    @property
    def has_minute(self) -> bool:
        """是否存在 minute 特徵流。"""
        return self.minute_feat > 0 and self.minute_idx_buf.numel() > 0

    def _split_by_indices(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        1. 說明: 依索引切分 base 與 minute 平面
        2. inputs:
           - x: [B, T, F]
        3. return:
           - (x_base, x_minute_flat): 兩者皆為 [B, T, feat] 或 None
        """
        x_base = x.index_select(-1, self.base_idx_buf) if self.base_idx_buf.numel() > 0 else None
        x_min  = x.index_select(-1, self.minute_idx_buf) if self.has_minute else None
        return x_base, x_min

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        1. 說明: 前向傳播；minute LSTM 與 base 特徵融合後送入 Transformer
        2. inputs:
           - x: [B, T, F]
           - key_padding_mask: [B, T]（True=遮蔽該 time step）
        3. return:
           - torch.Tensor: logits，形狀依 num_classes（=1 會在 backbone 內 squeeze）
        """
        B, T, F = x.shape
        x_base, x_min_flat = self._split_by_indices(x)

        # 無 minute ⇒ 直接單流
        if not self.has_minute or x_min_flat is None:
            if x_base is None:
                raise RuntimeError("無 minute 且 base_idx 為空；輸入全被吃掉了。請檢查切分。")
            if not self._shape_debug_printed:
                print(
                    f"[TwoStreamHybrid] x_base={tuple(x_base.shape)} | "
                    "x_min=None | "
                    f"fused={tuple(x_base.shape)}"
                )
                self._shape_debug_printed = True
            return self.backbone(x_base, key_padding_mask=key_padding_mask)

        # 還原 minute → [B*T, S, Fm]
        expected = self.minute_steps * self.minute_feat
        if x_min_flat.shape[-1] != expected:
            raise RuntimeError(f"minute 平面維度不符：got {x_min_flat.shape[-1]}, expected {expected}.")
        x_min = x_min_flat.reshape(B * T, self.minute_steps, self.minute_feat)

        # LSTMBlock（取最後一步）→ [B*T, H] → reshape 成 [B,T,H]
        micro_emb = self.min_block(x_min).reshape(B, T, -1)  # [B, T, H]

        # 與 base 拼接
        fused = micro_emb if x_base is None else torch.cat([x_base, micro_emb], dim=-1)  # [B,T, base_dim+H]
        if not self._shape_debug_printed:
            print(
                f"[TwoStreamHybrid] x_base={None if x_base is None else tuple(x_base.shape)} | "
                f"x_min={tuple(x_min.shape)} | "
                f"fused={tuple(fused.shape)}"
            )
            self._shape_debug_printed = True
        return self.backbone(fused, key_padding_mask=key_padding_mask)


@register("TwoStreamHybrid")
def build_two_stream_hybrid(cfg, n_features, columns):
    """
    1. 說明: 依 cfg 建立 TwoStreamHybrid，並掛入 registry
    2. inputs:
       - cfg: dict，需含 cfg["model"]（minute/backbone 超參、minute_prefixes 等）
       - n_features: int（未直接使用；實際以 columns 切分）
       - columns: list[str] 特徵欄位名，用於依 prefix 切 minute/base
    3. return:
       - nn.Module: TwoStreamHybrid
    """
    m = cfg["model"]
    return TwoStreamHybrid(
        # minute
        minute_steps   = m.get("minute_steps", 15),
        minute_hidden  = m.get("minute_hidden", 64),
        minute_layers  = m.get("minute_layers", 1),
        minute_dropout = m.get("minute_dropout", 0.0),
        # backbone
        d_model        = m.get("d_model", 128),
        n_heads        = m.get("n_heads", 4),
        num_layers     = m.get("num_layers", 2),
        mlp_ratio      = m.get("mlp_ratio", 4.0),
        dropout        = m.get("dropout", 0.1),
        attn_dropout   = m.get("attn_dropout", 0.0),
        pooling        = m.get("pooling", "attn"),
        use_causal     = m.get("use_causal", True),
        num_classes    = m.get("num_classes", 1),
        # 切分參數
        columns        = columns,
        minute_prefixes= tuple(m.get("minute_prefixes", ("m_",))),
    )
