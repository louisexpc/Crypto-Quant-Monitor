from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union, Dict

import torch
import torch.nn as nn


# ----------------------------------
# Transformer backbone（copied from train/models/transformer_model.py, trimmed for inference）
# ----------------------------------
def build_causal_mask(T: int, device, dtype=torch.bool) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, device=device, dtype=dtype), diagonal=1).bool()


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep * random_tensor


class SinCosPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        droppath: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.dp1 = DropPath(droppath)

        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.drop2 = nn.Dropout(dropout)
        self.dp2 = DropPath(droppath)

    def _forward_impl(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None) -> torch.Tensor:
        y = self.ln1(x)
        y, _ = self.attn(
            y, y, y,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dp1(self.drop1(y))

        y = self.ln2(x)
        y = self.mlp(y)
        x = x + self.dp2(self.drop2(y))
        return x

    def forward(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint
        return checkpoint(lambda _x: self._forward_impl(_x, attn_mask, key_padding_mask), x, use_reentrant=False)


class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q.expand(B, 1, D)
        scores = (q @ x.transpose(1, 2)) / math.sqrt(D)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ x
        return out.squeeze(1)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        pooling: str = "cls",
        use_causal: bool = False,
        droppath: float = 0.0,
    ):
        super().__init__()
        assert pooling in ("cls", "mean", "last", "attn")
        self.pooling = pooling
        self.use_causal = use_causal
        self.num_classes = num_classes

        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinCosPosEnc(d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, attn_dropout, dropout, droppath)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.attn_pool = AttnPool(d_model) if pooling == "attn" else None
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)

    def _build_attn_mask(self, T: int, has_cls: bool, device) -> Optional[torch.Tensor]:
        if not self.use_causal:
            return None
        mask = build_causal_mask(T, device=device, dtype=torch.bool)
        if has_cls:
            mask[0, :] = False
            mask[1:, 0] = True
        return mask

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.in_proj(x)
        x = self.pos_enc(x)

        has_cls = self.cls is not None
        if has_cls:
            cls_tok = self.cls.expand(B, -1, -1)
            x = torch.cat([cls_tok, x], dim=1)
            T = T + 1
            if key_padding_mask is not None:
                pad = torch.zeros((B, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)

        attn_mask = self._build_attn_mask(T=T, has_cls=has_cls, device=x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        if self.pooling == "cls":
            feat = x[:, 0]
        elif self.pooling == "last":
            feat = x[:, -1]
        elif self.pooling == "attn":
            feat = self.attn_pool(x, key_padding_mask=key_padding_mask)
        else:
            feat = x.mean(dim=1)

        logits = self.head(feat)
        return logits.squeeze(-1) if logits.size(-1) == 1 else logits


# ----------------------------------
# TwoStreamHybrid
# ----------------------------------
def build_feature_indices_by_prefix(
    columns: Optional[Sequence[str]],
    minute_prefixes: Tuple[str, ...] = ("m_",),
) -> Tuple[List[int], List[int]]:
    if not columns:
        return [], []
    if isinstance(minute_prefixes, str):
        minute_prefixes = (minute_prefixes,)
    minute_idx = [i for i, c in enumerate(columns) if c.startswith(minute_prefixes)]
    base_idx = [i for i, c in enumerate(columns) if not c.startswith(minute_prefixes)]
    return minute_idx, base_idx


class LSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
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
        y, _ = self.lstm(x)
        return y[:, -1, :]


class TwoStreamHybrid(nn.Module):
    def __init__(
        self,
        minute_steps: int = 15,
        minute_hidden: int = 64,
        minute_layers: int = 1,
        minute_dropout: float = 0.0,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        pooling: str = "attn",
        use_causal: bool = True,
        num_classes: int = 1,
        columns: Optional[Sequence[str]] = None,
        minute_prefixes: Tuple[str, ...] = ("m_",),
    ):
        super().__init__()
        minute_idx, base_idx = build_feature_indices_by_prefix(columns, minute_prefixes)
        self.base_dim = len(base_idx)

        if len(minute_idx) > 0:
            if len(minute_idx) % minute_steps != 0:
                raise ValueError(f"len(minute_idx)={len(minute_idx)} cannot be divided by minute_steps={minute_steps}")
            minute_feat = len(minute_idx) // minute_steps
        else:
            minute_feat = 0

        self.minute_steps = int(minute_steps)
        self.minute_feat = int(minute_feat)

        self.register_buffer("minute_idx_buf", torch.as_tensor(minute_idx, dtype=torch.long), persistent=False)
        self.register_buffer("base_idx_buf", torch.as_tensor(base_idx, dtype=torch.long), persistent=False)

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

        in_dim_backbone = self.base_dim + self.min_out_dim
        if in_dim_backbone == 0:
            raise ValueError("Backbone input dim is 0; check feature columns/prefix.")

        self.backbone = TemporalTransformer(
            input_dim=in_dim_backbone,
            num_classes=num_classes,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            pooling=pooling,
            use_causal=use_causal,
        )

    @property
    def has_minute(self) -> bool:
        return self.minute_feat > 0 and self.minute_idx_buf.numel() > 0

    def _split_by_indices(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        x_base = x.index_select(-1, self.base_idx_buf) if self.base_idx_buf.numel() > 0 else None
        x_min = x.index_select(-1, self.minute_idx_buf) if self.has_minute else None
        return x_base, x_min

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        B, T, F = x.shape
        x_base, x_min_flat = self._split_by_indices(x)

        if not self.has_minute or x_min_flat is None:
            if x_base is None:
                raise RuntimeError("No base or minute features found; check columns/prefix.")
            return self.backbone(x_base, key_padding_mask=key_padding_mask)

        expected = self.minute_steps * self.minute_feat
        if x_min_flat.shape[-1] != expected:
            raise RuntimeError(f"minute dim mismatch: got {x_min_flat.shape[-1]}, expected {expected}.")
        x_min = x_min_flat.reshape(B * T, self.minute_steps, self.minute_feat)
        micro_emb = self.min_block(x_min).reshape(B, T, -1)
        fused = micro_emb if x_base is None else torch.cat([x_base, micro_emb], dim=-1)
        return self.backbone(fused, key_padding_mask=key_padding_mask)


# ----------------------------------
# build_model
# ----------------------------------
def build_model(model_cfg: Dict[str, Any], n_features: int, columns: Sequence[str]) -> nn.Module:
    """
    1. 說明: 根據 model_cfg.name 建模；支援 TwoStreamHybrid / TemporalTransformer。
    2. inputs:
        - model_cfg: dict，需含 name
        - n_features: 特徵數（若有 minute 前綴會依 columns 切分）
        - columns: 特徵欄位名
    3. return:
        - nn.Module
    """
    name = str(model_cfg.get("name", "")).lower()
    if name == "twostreamhybrid":
        return TwoStreamHybrid(
            minute_steps=model_cfg.get("minute_steps", 15),
            minute_hidden=model_cfg.get("minute_hidden", 64),
            minute_layers=model_cfg.get("minute_layers", 1),
            minute_dropout=model_cfg.get("minute_dropout", 0.0),
            d_model=model_cfg.get("d_model", 128),
            n_heads=model_cfg.get("n_heads", 4),
            num_layers=model_cfg.get("n_layers", model_cfg.get("num_layers", 2)),
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            dropout=model_cfg.get("dropout", 0.1),
            attn_dropout=model_cfg.get("attn_dropout", 0.0),
            pooling=model_cfg.get("pooling", "attn"),
            use_causal=model_cfg.get("use_causal", True),
            num_classes=model_cfg.get("num_classes", 1),
            columns=columns,
            minute_prefixes=tuple(model_cfg.get("minute_prefixes", ("m_",))),
        )
    if name == "temporaltransformer":
        return TemporalTransformer(
            input_dim=n_features,
            num_classes=model_cfg.get("num_classes", 1),
            d_model=model_cfg.get("d_model", 128),
            n_heads=model_cfg.get("n_heads", 4),
            num_layers=model_cfg.get("n_layers", model_cfg.get("num_layers", 3)),
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            dropout=model_cfg.get("dropout", 0.1),
            attn_dropout=model_cfg.get("attn_dropout", 0.0),
            pooling=model_cfg.get("pooling", "cls"),
            use_causal=model_cfg.get("use_causal", False),
            droppath=model_cfg.get("droppath", 0.0),
        )
    raise ValueError(f"Unsupported model: {model_cfg.get('name')}")


__all__ = [
    "TwoStreamHybrid",
    "TemporalTransformer",
    "build_model",
]
