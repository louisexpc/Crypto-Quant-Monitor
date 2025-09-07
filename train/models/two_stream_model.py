# two_stream_model.py
import torch
import torch.nn as nn
from typing import Sequence, Optional, Tuple
from .Transformer import TemporalTransformer  # 直接沿用你現有的主幹


def build_feature_indices_by_prefix(
    columns: Sequence[str],
    minute_prefixes: Tuple[str, ...] = ("m_",),
) -> Tuple[list[int], list[int]]:
    """
    根據欄名 prefix 切分 minute 與 base 的欄位索引。
    - minute_prefixes: 可傳單一字串或多個前綴的 tuple
    """
    if isinstance(minute_prefixes, str):
        minute_prefixes = (minute_prefixes,)
    minute_idx = [i for i, c in enumerate(columns) if c.startswith(minute_prefixes)]
    base_idx   = [i for i, c in enumerate(columns) if not c.startswith(minute_prefixes)]
    return minute_idx, base_idx




class LSTMBlock(nn.Module):
    """
    可重用的 LSTM 子模組（序列 -> 向量）。
    預設行為與原 TwoStreamHybrid 的分鐘流一致：單向、多層可選、取最後一步作為輸出。

    Args:
        input_size:   每時間步輸入維度
        hidden_size:  LSTM 隱層（單向）
        num_layers:   LSTM 層數
        dropout:      層間 dropout（僅 num_layers > 1 時生效）

    Inputs:
        x: [B, S, F] 序列張量

    Returns:
        out: [B, hidden_size] 向量（取最後一步）
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # ★ 與原實作一致：單向
        )
        self.out_dim = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, F]
        y, _ = self.lstm(x)     # [B, S, H]
        out = y[:, -1, :]       # ★ 與原實作一致：取最後一步
        return out


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

        # 以欄名切分所需參數（二擇一）
        columns: Optional[Sequence[str]] = None,
        minute_prefixes: Tuple[str, ...] = ("m_",)
    ):
        super().__init__()

        # ---- 1) 建立 minute/base 索引（優先用 columns+prefix；或用使用者給定的 idx）----
        mi, ba = build_feature_indices_by_prefix(columns, minute_prefixes)     
        minute_idx = list(mi)
        base_idx   = list(ba)

        # ---- 2) 由 minute_idx 推斷 minute_feat（不再當成參數）----
        if len(minute_idx) > 0:
            if len(minute_idx) % minute_steps != 0:
                raise ValueError(f"len(minute_idx)={len(minute_idx)} 不能被 minute_steps={minute_steps} 整除；無法推斷 minute_feat。")
            minute_feat = len(minute_idx) // minute_steps
        else:
            minute_feat = 0  # 無 minute 特徵

        self.minute_steps = int(minute_steps)
        self.minute_feat  = int(minute_feat)   # 僅作內部形狀檢查
        self.base_dim     = len(base_idx)

        # 註冊索引為 buffer（隨 .to(device) 移動）
        self.register_buffer("minute_idx_buf", torch.as_tensor(minute_idx, dtype=torch.long), persistent=False)
        self.register_buffer("base_idx_buf",   torch.as_tensor(base_idx,   dtype=torch.long), persistent=False)

        # ---- 3) minute encoder：固定「單向」LSTM（若無 minute 特徵則略過）----
        if self.minute_feat > 0:
            self.min_block = LSTMBlock(
                input_size=self.minute_feat,
                hidden_size=minute_hidden,
                num_layers=minute_layers,
                dropout=minute_dropout,
            )
            self.min_out_dim = self.min_block.out_dim
        else:
            self.lstm = None
            self.min_out_dim = 0

        # ---- 4) Backbone ----
        in_dim_backbone = self.base_dim + self.min_out_dim if self.min_out_dim > 0 else self.base_dim
        if in_dim_backbone == 0:
            raise ValueError("in_dim_backbone = 0：看起來既沒有 base 也沒有 minute 特徵。請檢查 columns/prefix 或索引。")

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

    @property
    def has_minute(self) -> bool:
        return self.minute_feat > 0 and self.minute_idx_buf.numel() > 0

    def _split_by_indices(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """回傳 (x_base, x_minute_flat)，皆為 [B,T,feat]。"""
        if self.base_idx_buf.numel() > 0:
            x_base = x.index_select(-1, self.base_idx_buf)
        else:
            x_base = None

        if self.has_minute:
            x_min_flat = x.index_select(-1, self.minute_idx_buf)
        else:
            x_min_flat = None

        return x_base, x_min_flat

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        B, T, F = x.shape
        x_base, x_min_flat = self._split_by_indices(x)

        # 無 minute ⇒ 直接走單流
        if not self.has_minute or x_min_flat is None:
            if x_base is None:
                raise RuntimeError("無 minute 且 base_idx 為空，輸入全被吃掉了。請檢查切分。")
            return self.backbone(x_base, key_padding_mask=key_padding_mask)

        # 還原 minute → [B*T, S, Fm]
        expected = self.minute_steps * self.minute_feat
        if x_min_flat.shape[-1] != expected:
            raise RuntimeError(f"minute 平面維度不符：got {x_min_flat.shape[-1]}, expected {expected}.")
        x_min = x_min_flat.reshape(B * T, self.minute_steps, self.minute_feat)

        # LSTMBlock 取最後一步）→ [B*T, H] → reshape 成 [B,T,H]
        # y, _ = self.lstm(x_min)                  # [B*T, S, H]
        # micro_emb = y[:, -1, :].reshape(B, T, -1)  # [B, T, H]
        micro_emb = self.min_block(x_min).reshape(B, T, -1)  # [B,T,H]

        # 拼接
        fused = micro_emb if x_base is None else torch.cat([x_base, micro_emb], dim=-1)  # [B,T, base_dim+H]
        return self.backbone(fused, key_padding_mask=key_padding_mask)