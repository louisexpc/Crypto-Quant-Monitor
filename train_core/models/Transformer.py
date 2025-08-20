# models/transformer_model.py

import math
import torch
import torch.nn as nn

# ---------- 小工具 ----------
def build_causal_mask(T: int, device, dtype=torch.float32):
    # [T,T] 上三角為 -inf，禁止注意未來
    mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)

def build_alibi_bias(T: int, device, slope: float = 0.05, dtype=torch.float32):
    # 簡化版 ALiBi：距離越遠，懲罰越大（負向）
    # bias[i,j] = -slope * max(0, j-i)
    i = torch.arange(T, device=device).unsqueeze(1)
    j = torch.arange(T, device=device).unsqueeze(0)
    dist = (j - i).clamp_min(0).to(dtype)
    return -slope * dist  # [T,T], 可與 causal mask 相加

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
        y, _ = self.attn(y, y, y, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dp1(self.drop1(y))
        # MLP（Pre-Norm）
        y = self.ln2(x)
        y = self.mlp(y)
        x = x + self.dp2(self.drop2(y))
        return x

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(lambda _x: self._forward_impl(_x, attn_mask, key_padding_mask), x, use_reentrant=False)

# ---- 3) 注意力池化（比 mean/last 更靈活）----
class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)  # learnable query
    def forward(self, x, key_padding_mask=None):
        B, T, D = x.shape
        q = self.q.expand(B, -1, -1)                     # [B,1,D]
        attn = torch.softmax((q @ x.transpose(1, 2)) / math.sqrt(D), dim=-1)  # [B,1,T]
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(1).float()  # True=keep
            attn = attn * mask
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
        out = attn @ x                                    # [B,1,D]
        return out.squeeze(1)                             # [B,D]

# ---- 4) 可選：深度可分離一維卷積（局部多尺度）----
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
                 ):
        super().__init__()
        self.pooling = pooling
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

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, F]
        B, T, _ = x.shape
        x = self.in_proj(x)           # [B, T, D]
        x = self.pos_enc(x)           # 加上位置編碼

        if self.cls is not None:
            cls_tok = self.cls.expand(B, -1, -1)
            x = torch.cat([cls_tok, x], dim=1)   # [B, T+1, D]
            T += 1  # 增加 token 數
            if key_padding_mask is not None:
                pad = torch.zeros((B, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
                key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)
       
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()  # [T, T]
        
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        x = self.norm(x)

        # Pooling
        if self.pooling == "cls":
            feat = x[:, 0]
        elif self.pooling == "last":
            feat = x[:, -1]
        elif self.pooling == "attn":
            feat = self.attn_pool(x, key_padding_mask=key_padding_mask)
        else:
            feat = x.mean(dim=1)
        # print(f"[DEBUG] feat.mean = {feat.mean().item():.6f}, std = {feat.std().item():.6f}")
        return self.head(feat)
