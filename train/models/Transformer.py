# train/models/transformer_model.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
from train.models.registry import register


def build_causal_mask(T: int, device, dtype=torch.bool):
    """
    1. 說明: 產生標準因果遮罩，上三角(含對角線以上)為 True 表示遮蔽未來。
    2. inputs:
       - T: 序列長度
       - device: torch 裝置
       - dtype: 遮罩 dtype（預設 bool）
    3. return:
       - torch.Tensor: [T, T] 的遮罩矩陣（True=mask 掉）
    """
    return torch.triu(torch.ones(T, T, device=device, dtype=dtype), diagonal=1).bool()


class DropPath(nn.Module):
    """Stochastic Depth：按樣本隨機丟掉殘差支路。"""
    def __init__(self, drop_prob: float = 0.0):
        """
        1. 說明: 初始化 DropPath
        2. inputs:
           - drop_prob: 丟棄機率（0~1）
        3. return: None
        """
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. 說明: 訓練時隨機丟棄殘差；推論時直接通過
        2. inputs:
           - x: 輸入張量
        3. return:
           - torch.Tensor: 與 x 同形狀
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 0/1
        return x / keep * random_tensor


class SinCosPosEnc(nn.Module):
    """固定正弦/餘弦位置編碼（不存入 checkpoint）。"""
    def __init__(self, d_model: int, max_len: int = 4096):
        """
        1. 說明: 建立 Sin/Cos 位置編碼表
        2. inputs:
           - d_model: 特徵維度
           - max_len: 支援的最長序列
        3. return: None
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [L,1]
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. 說明: 對輸入加上位置編碼
        2. inputs:
           - x: [B, T, D]
        3. return:
           - torch.Tensor: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # [1,T,D] + [B,T,D]


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer block（MHSA + MLP），含 DropPath。"""
    def __init__(
        self, d_model: int, n_heads: int, mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0, dropout: float = 0.1, droppath: float = 0.0
    ):
        """
        1. 說明: 初始化單一 Transformer 區塊
        2. inputs:
           - d_model: 模型維度
           - n_heads: 注意力頭數
           - mlp_ratio: MLP 擴張倍數
           - attn_dropout: 注意力 dropout
           - dropout: 一般 dropout
           - droppath: DropPath 機率
        3. return: None
        """
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
        """
        1. 說明: block 內部實作（供 checkpoint 包裝）
        2. inputs:
           - x: [B, T, D]
           - attn_mask: [T, T]（bool/float，True 或 -inf 表示遮蔽）
           - key_padding_mask: [B, T]（bool，True=遮蔽該 time step）
        3. return:
           - torch.Tensor: [B, T, D]
        """
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
        """
        1. 說明: 前向；用 checkpoint 降顯存
        2. inputs:
           - x: [B, T, D]
           - attn_mask: [T, T]
           - key_padding_mask: [B, T]
        3. return:
           - torch.Tensor: [B, T, D]
        """
        from torch.utils.checkpoint import checkpoint
        return checkpoint(lambda _x: self._forward_impl(_x, attn_mask, key_padding_mask), x, use_reentrant=False)


class AttnPool(nn.Module):
    """單查詢的注意力池化（learnable query）。"""
    def __init__(self, d_model: int):
        """
        1. 說明: 初始化注意力池化
        2. inputs:
           - d_model: 內部維度
        3. return: None
        """
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        """
        1. 說明: 將 [B,T,D] 池化成 [B,D]
        2. inputs:
           - x: [B, T, D]
           - key_padding_mask: [B, T]（True=遮蔽）
        3. return:
           - torch.Tensor: [B, D]
        """
        B, T, D = x.shape
        q = self.q.expand(B, 1, D)                      # [B,1,D]
        scores = (q @ x.transpose(1, 2)) / math.sqrt(D) # [B,1,T]
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)            # [B,1,T]
        out = attn @ x                                  # [B,1,D]
        return out.squeeze(1)                           # [B,D]


class ConvStem1D(nn.Module):
    """可選的 1D 深/逐層卷積前端（預設不啟用）。"""
    def __init__(self, d_model: int, k: int = 5, pw: bool = True):
        """
        1. 說明: 初始化 1D 卷積前端
        2. inputs:
           - d_model: 通道數
           - k: kernel size
           - pw: 是否加 point-wise conv
        3. return: None
        """
        super().__init__()
        pad = k // 2
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1) if pw else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. 說明: 1D depthwise + pointwise
        2. inputs:
           - x: [B, T, D]
        3. return:
           - torch.Tensor: [B, T, D]
        """
        x = x.transpose(1, 2)           # [B,D,T]
        x = self.pw(self.act(self.dw(x)))
        return x.transpose(1, 2)        # [B,T,D]


class TemporalTransformer(nn.Module):
    """時間序列 Transformer（支持 CLS/mean/last/attention 池化與可選因果遮罩）。"""
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
        pooling: str = "cls",       # ["cls", "mean", "last", "attn"]
        use_causal: bool = False,   # 是否使用因果遮罩
        droppath: float = 0.0,      # Stochastic Depth 機率（每層同值）
    ):
        """
        1. 說明: 初始化 TemporalTransformer
        2. inputs:
           - input_dim: 輸入特徵維度 F
           - num_classes: 類別數（=1 視為二分類/回歸）
           - d_model: 模型維度
           - n_heads: 注意力頭數
           - num_layers: block 數
           - mlp_ratio/dropout/attn_dropout/droppath: 超參
           - pooling: 聚合策略
           - use_causal: 是否套用因果遮罩
        3. return: None
        """
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

    def _build_attn_mask(self, T: int, has_cls: bool, device) -> torch.Tensor | None:
        """
        1. 說明: 產生 self-attn 遮罩（bool，True=遮蔽）
        2. inputs:
           - T: 現在序列長度（若有 CLS 已含在內）
           - has_cls: 是否使用 CLS token
           - device: torch 裝置
        3. return:
           - torch.Tensor|None: [T, T] 遮罩或 None
        """
        if not self.use_causal:
            return None

        mask = build_causal_mask(T, device=device, dtype=torch.bool)  # 上三角 True

        if has_cls:
            # 讓 CLS（row 0）可看所有 token
            mask[0, :] = False
            # 其他 token 禁止看 CLS（避免透過 CLS 間接偷看未來）
            mask[1:, 0] = True

        return mask

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        1. 說明: 前向傳播；輸入 [B,T,F]，輸出 logits
        2. inputs:
           - x: [B, T, F]
           - key_padding_mask: [B, T]（True=遮蔽該 time step）
        3. return:
           - torch.Tensor: [B, C]；若 C=1 則回傳 [B]
        """
        B, T, _ = x.shape
        x = self.in_proj(x)           # [B, T, D]
        x = self.pos_enc(x)           # 位置編碼

        has_cls = self.cls is not None
        if has_cls:
            cls_tok = self.cls.expand(B, -1, -1)           # [B,1,D]
            x = torch.cat([cls_tok, x], dim=1)             # [B, T+1, D]
            T = T + 1
            if key_padding_mask is not None:
                pad = torch.zeros((B, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)  # CLS 永不遮蔽

        attn_mask = self._build_attn_mask(T=T, has_cls=has_cls, device=x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        x = self.norm(x)

        # 聚合
        if self.pooling == "cls":
            feat = x[:, 0]                         # CLS
        elif self.pooling == "last":
            feat = x[:, -1]
        elif self.pooling == "attn":
            feat = self.attn_pool(x, key_padding_mask=key_padding_mask)
        else:  # "mean"
            feat = x.mean(dim=1)

        logits = self.head(feat)                   # [B, C]
        return logits.squeeze(-1) if logits.size(-1) == 1 else logits


@register("TemporalTransformer")
def build_temporal_transformer(cfg, n_features, columns):
    """
    1. 說明: 依 cfg 建立 TemporalTransformer，並掛入 registry
    2. inputs:
       - cfg: dict（需含 cfg["model"]：d_model/n_heads/num_layers/...）
       - n_features: int，輸入特徵數
       - columns: list[str]，特徵名（此處未使用，為介面對齊預留）
    3. return:
       - nn.Module: TemporalTransformer
    """
    m = cfg["model"]
    return TemporalTransformer(
        input_dim     = n_features,
        num_classes   = m.get("num_classes", 1),
        d_model       = m.get("d_model", 128),
        n_heads       = m.get("n_heads", 4),
        num_layers    = m.get("num_layers", 3),
        mlp_ratio     = m.get("mlp_ratio", 4.0),
        dropout       = m.get("dropout", 0.1),
        attn_dropout  = m.get("attn_dropout", 0.0),
        pooling       = m.get("pooling", "cls"),
        use_causal    = m.get("use_causal", False),
        droppath      = m.get("droppath", 0.0),
    )
