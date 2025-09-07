# models/transformer_model.py

import math
import torch
import torch.nn as nn

# ---------- 小工具 ----------
def build_causal_mask(T: int, device, dtype=torch.bool):
    """
    標準因果遮罩：上三角(對角線以上)為 True 表示要遮蔽。
    回傳 [T, T] bool。
    """
    return torch.triu(torch.ones(T, T, device=device, dtype=dtype), diagonal=1).bool()

class DropPath(nn.Module):
    """Stochastic Depth（按樣本隨機丟掉殘差）。"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 0/1
        return x / keep * random_tensor

# ---- 1) Sin-Cos 位置編碼 ----
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
                 attn_dropout: float = 0.0, dropout: float = 0.1, droppath: float = 0.0):
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

    def _forward_impl(self, x, attn_mask=None, key_padding_mask=None):
        # 注意力（Pre-Norm）
        y = self.ln1(x)
        y, _ = self.attn(
            y, y, y,
            attn_mask=attn_mask,                 # [T,T] bool 或 float
            key_padding_mask=key_padding_mask,   # [B,T] bool（True=要遮）
            need_weights=False
        )
        x = x + self.dp1(self.drop1(y))
        # MLP（Pre-Norm）
        y = self.ln2(x)
        y = self.mlp(y)
        x = x + self.dp2(self.drop2(y))
        return x

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(lambda _x: self._forward_impl(_x, attn_mask, key_padding_mask), x, use_reentrant=False)

# ---- 3) 注意力池化（mask 在 softmax 前套用）----
class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)  # learnable query
    def forward(self, x, key_padding_mask=None):
        # x: [B,T,D]; key_padding_mask: [B,T]（True=要遮）
        B, T, D = x.shape
        q = self.q.expand(B, -1, -1)                                  # [B,1,D]
        scores = (q @ x.transpose(1, 2)) / math.sqrt(D)               # [B,1,T]
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)                          # [B,1,T]
        out = attn @ x                                                # [B,1,D]
        return out.squeeze(1)                                         # [B,D]

# ---- 4)（可選）ConvStem：局部多尺度 ----
class ConvStem1D(nn.Module):
    def __init__(self, d_model: int, k: int = 5, pw: bool = True):
        super().__init__()
        pad = k // 2
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1) if pw else nn.Identity()
        self.act = nn.GELU()
    def forward(self, x):   # x: [B,T,D]
        x = x.transpose(1, 2)           # [B,D,T]
        x = self.pw(self.act(self.dw(x)))
        return x.transpose(1, 2)        # [B,T,D]

# ---- 5) 主模型 ----
class TemporalTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_classes: int = 1,
                 d_model: int = 128,
                 n_heads: int = 4,
                 num_layers: int = 3,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.0,
                 pooling: str = "cls",       # ["cls", "mean", "last", "attn"]
                 use_causal: bool = False,   # ← 新增：是否使用因果遮罩
                 ):
        super().__init__()
        assert pooling in ("cls", "mean", "last", "attn")
        self.pooling = pooling
        self.use_causal = use_causal

        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinCosPosEnc(d_model)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, attn_dropout, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.attn_pool = AttnPool(d_model) if pooling == "attn" else None
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)

    def _build_attn_mask(self, T: int, has_cls: bool, device):
        """
        產生 self-attn 的 attn_mask（bool，True=遮蔽）。
        - use_causal=False → None（不遮）
        - use_causal=True  →
            * 無 CLS：標準 causal（上三角 True）
            * 有 CLS（在 idx=0）：
                - row 0（CLS 查詢）可看所有 token（行 0 全 False）
                - 其他行：禁止看未來 + 禁止看 CLS（col 0 True）
        """
        if not self.use_causal:
            return None

        mask = build_causal_mask(T, device=device, dtype=torch.bool)  # 上三角 True

        if has_cls:
            # 讓 CLS（row 0）能看所有（覆寫掉上三角）
            mask[0, :] = False
            # 禁止其他 token 看 CLS（避免經由 CLS 間接獲得未來訊息）
            mask[1:, 0] = True

        return mask

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, F]; key_padding_mask: [B, T]（True=要遮）
        B, T, _ = x.shape
        x = self.in_proj(x)           # [B, T, D]
        x = self.pos_enc(x)           # 加上位置編碼

        has_cls = self.cls is not None
        if has_cls:
            cls_tok = self.cls.expand(B, -1, -1)          # [B,1,D]
            x = torch.cat([cls_tok, x], dim=1)            # [B, T+1, D]
            T = T + 1
            if key_padding_mask is not None:
                pad = torch.zeros((B, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)  # CLS 不遮蔽

        attn_mask = self._build_attn_mask(T=T, has_cls=has_cls, device=x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        x = self.norm(x)

        # Pooling
        if self.pooling == "cls":
            feat = x[:, 0]                         # CLS（最前）
        elif self.pooling == "last":
            feat = x[:, -1]
        elif self.pooling == "attn":
            feat = self.attn_pool(x, key_padding_mask=key_padding_mask)
        else:  # "mean"
            feat = x.mean(dim=1)

        return self.head(feat)
