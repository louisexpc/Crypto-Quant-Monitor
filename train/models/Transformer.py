# models/transformer_model.py

import math
import torch
import torch.nn as nn

# ---- 1) Sin-Cos 位置編碼（穩定好用）----
class SinCosPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [L,1]
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)  # 不存入 ckpt

    def forward(self, x):  # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)  # [1,T,D] + [B,T,D]


# ---- 2) Transformer Block：Pre-Norm + 殘差 ----
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def _forward_impl(self, x, key_padding_mask=None):
        # x: [B, T, D]

        # ① 注意力支路（Pre-Norm）
        # Pre-Norm 殘差：x = x + Attn(LN(x))
        y = self.ln1(x) # 先做 LayerNorm，穩定分佈，便於後續模組學習
        y, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(y) # 殘差：把注意力輸出疊回原輸入

        # ② MLP 支路（Pre-Norm）
        y = self.ln2(x)
        y = self.mlp(y) # Linear→GELU→Dropout→Linear
        x = x + self.drop2(y) # 殘差疊回
        return x
    
    def forward(self, x, key_padding_mask=None):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(lambda _x: self._forward_impl(_x, key_padding_mask), x, use_reentrant=False)
    
# ---- 3) Temporal-only Transformer 分類器 ----
class TemporalTransformer(nn.Module):
    """
    輸入:  x ∈ [B, T, F]
    輸出:  logits ∈ [B, C]
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int = 2,
                 d_model: int = 128,
                 n_heads: int = 4,
                 num_layers: int = 3,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.0,
                 pooling: str = "cls",       # ["cls", "mean", "last"]
                 use_learned_pos: bool = False):
        super().__init__()
        self.pooling = pooling

        # 將原始特徵 F 投影到 d_model
        self.in_proj = nn.Linear(input_dim, d_model)

        # 位置編碼（預設 sin-cos；也可切換 learned pos）
        self.pos_enc = (nn.Embedding(4096, d_model) if use_learned_pos else SinCosPosEnc(d_model))

        # 可選 CLS token（做全局摘要）
        if pooling == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls = None

        # 堆疊 Transformer blocks（Pre-Norm + 殘差）
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, attn_dropout, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        # 參數初始化（Kaiming/小幅度）
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x, key_padding_mask=None):
        # x: [B,T,F] → [B,T,D]
        x = self.in_proj(x)

        # 位置編碼
        if isinstance(self.pos_enc, SinCosPosEnc):
            x = self.pos_enc(x)
        else:  # learned pos
            B, T, D = x.shape
            pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            x = x + self.pos_enc(pos_ids)

        # 若使用 CLS，前置到序列開頭
        if self.cls is not None:
            B = x.size(0)
            cls_tok = self.cls.expand(B, -1, -1)   # [B,1,D]
            x = torch.cat([cls_tok, x], dim=1)     # [B,1+T,D]
            # key_padding_mask 也要對齊（若有提供）
            if key_padding_mask is not None:
                pad = torch.zeros((B, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)

        # Transformer 堆疊
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)

        # 池化成單一向量
        if self.pooling == "cls":
            feat = x[:, 0]                 # CLS 向量
        elif self.pooling == "last":
            feat = x[:, -1]                # 最後時間步
        else:  # "mean"
            # 若有 padding，可用 (1 - key_padding_mask) 做加權平均；這裡假設定長序列
            feat = x.mean(dim=1)

        logits = self.head(feat)           # [B,C]
        return logits

