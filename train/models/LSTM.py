import os, torch, torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_SE(nn.Module):
    """
    LSTM → (B,T,C) ─► SE(通道重標定) ─► 池化(mean+max) ─► FC
    x: [B, T, F] -> logits: [B, C]
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        se_ratio: int = 16,
        num_classes: int = 3,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        # 1. layer_norm: 讓每個 feature 維度分佈更穩
        self.ln_in = nn.LayerNorm(input_dim)

        # 2. LSTM: 輸出 [B, T, H]
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1) 


        # Squeeze-and-Excitation (通道注意力)
        se_hidden = max(1, out_dim // se_ratio)            # 避免太小時變 0
        self.se = nn.Sequential(
            nn.Linear(out_dim, se_hidden),
            nn.ReLU(),
            nn.Linear(se_hidden, out_dim),
            nn.Sigmoid(),
        )

        self.ln_post = nn.LayerNorm(out_dim * 2)  # mean+max concat
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(out_dim * 2, out_dim)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]

        x = self.ln_in(x)
        y, _ = self.lstm(x)                  # [B, T, H*dir]

        # 時序池化: mean + max
        mean_pool = y.mean(dim=1)            # [B, H*dir]
        max_pool  = y.max(dim=1).values      # [B, H*dir]
        pooled    = (mean_pool + max_pool) / 2.0  # 先平均做 SE 比較穩

        # SE 通道重加權
        w = self.se(pooled)                  # [B, H*dir]
        y_last = y[:, -1, :] * w             # 用最後步並乘上權重（也可改成 pooled * w）

        # head
        feat = torch.cat([mean_pool, max_pool], dim=1)  # [B, 2*H*dir]
        feat = self.ln_post(feat)
        feat = self.dropout(F.relu(self.hidden(feat)))  # [B, H*dir]
        logits = self.fc(feat)                          # [B, C]
        return logits.squeeze(-1)        # [B] → 和 y_true 一致





class TimeAttention(nn.Module):
    """單頭線性打分：score_t = w_a^T h_t；回傳 context 與注意力權重 alpha"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.wa = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H: torch.Tensor):
        # H: [B, T, H]
        scores = self.wa(H).squeeze(-1)   # [B, T]
        alpha  = scores.softmax(dim=1)    # [B, T]
        context = (alpha.unsqueeze(-1) * H).sum(dim=1)  # [B, H]
        return context, alpha

class LSTM_Temporal(nn.Module):
    """
    LSTM → Attention → (可選 SE) → (可選 mean/max Pooling 融合) → FC → logits[B, C]
    透過三個旗標對應三種設定：
      - use_se=False,  pooling='none'       → ATTN
      - use_se=True,   pooling='none'       → ATTN+SE
      - use_se=True,   pooling='meanmax'    → ATTN+SE+POOLING
    你也可以試 use_se=False, pooling='meanmax'（ATTN+POOLING），方便做完整消融。
    """
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        num_classes: int = 3,
        # 三個控制點
        use_se: bool = False,
        se_ratio: int = 16,
        pooling: str = "none",   # 'none' or 'meanmax'
        return_alpha: bool = False,
    ):
        super().__init__()
        assert pooling in ("none", "meanmax")
        self.bidirectional = bidirectional
        self.use_se = use_se
        self.pooling = pooling
        self.return_alpha = return_alpha

        self.ln_in = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        H = hidden_size * (2 if bidirectional else 1)

        # 時序 Attention（論文式）
        self.attn = TimeAttention(H)

        # 通道 SE（對 context/池化向量做門控）
        if use_se:
            se_hidden = max(1, H // se_ratio)
            self.se = nn.Sequential(
                nn.Linear(H, se_hidden),
                nn.ReLU(),
                nn.Linear(se_hidden, H),
                nn.Sigmoid(),
            )

        # head 輸入維度：只有 context → H；加 mean/max → 3H
        head_in = H if pooling == "none" else (H * 3)
        self.ln_post = nn.LayerNorm(head_in)
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(head_in, H)
        self.fc = nn.Linear(H, num_classes)

    def forward(self, x: torch.Tensor):
        # x: [B, T, F]
        x = self.ln_in(x)
        Y, _ = self.lstm(x)                       # [B, T, H]

        # 時序 Attention
        context, alpha = self.attn(Y)              # [B, H], [B, T]

        # （可選）時序池化分支
        if self.pooling == "meanmax":
            mean_pool = Y.mean(dim=1)             # [B, H]
            max_pool  = Y.amax(dim=1)             # [B, H]

        # （可選）SE 通道門控：以更穩的摘要作 gate 的輸入
        if self.use_se:
            if self.pooling == "meanmax":
                gate_in = 0.5 * (mean_pool + max_pool)  # 比單獨 context 更穩
            else:
                gate_in = context
            w = self.se(gate_in)                   # [B, H]
            context = context * w
            if self.pooling == "meanmax":
                mean_pool = mean_pool * w
                max_pool  = max_pool  * w

        # head 特徵拼接
        if self.pooling == "meanmax":
            feat = torch.cat([context, mean_pool, max_pool], dim=1)   # [B, 3H]
        else:
            feat = context                                            # [B, H]

        feat = self.ln_post(feat)
        feat = self.dropout(F.relu(self.hidden(feat)))                 # [B, H]
        logits = self.fc(feat)                                         # [B, C]

        if self.return_alpha:
            return logits, alpha
        return logits