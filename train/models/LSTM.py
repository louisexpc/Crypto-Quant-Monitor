from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from train.models.registry import register


class LSTM_SE(nn.Module):
    """
    LSTM → (B,T,C) → SE(通道重標定) → 池化(mean+max) → FC
    x: [B, T, F] → logits: [B, C]
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_size: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        se_ratio: int = 16,
        num_classes: int = 2,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        # 1) 輸入正規化
        self.ln_in = nn.LayerNorm(input_dim)

        # 2) LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)

        # 3) Squeeze-and-Excitation（通道注意力）
        se_hidden = max(1, out_dim // se_ratio)
        self.se = nn.Sequential(
            nn.Linear(out_dim, se_hidden),
            nn.ReLU(),
            nn.Linear(se_hidden, out_dim),
            nn.Sigmoid(),
        )

        # 4) Head
        self.ln_post = nn.LayerNorm(out_dim * 2)  # mean+max concat
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(out_dim * 2, out_dim)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. 說明: 前向傳播；回傳 logits，形狀隨 num_classes 決定
        2. inputs:
           - x: [B, T, F]
        3. return:
           - torch.Tensor: [B, C]；若 C=1 則回傳 [B]
        """
        x = self.ln_in(x)
        y, _ = self.lstm(x)                     # [B, T, H*dir]

        # 時序池化
        mean_pool = y.mean(dim=1)               # [B, H*dir]
        max_pool  = y.amax(dim=1)               # [B, H*dir]

        # 用 mean+max 的摘要產生 SE 權重，並作用回兩個池化向量（更穩定）
        pooled = 0.5 * (mean_pool + max_pool)   # [B, H*dir]
        w = self.se(pooled)                     # [B, H*dir]
        mean_pool = mean_pool * w
        max_pool  = max_pool  * w

        # head
        feat = torch.cat([mean_pool, max_pool], dim=1)  # [B, 2H*dir]
        feat = self.ln_post(feat)
        feat = self.dropout(F.relu(self.hidden(feat)))  # [B, H*dir]
        logits = self.fc(feat)                          # [B, C]

        # 只有二分類（C=1）才 squeeze；多分類保持 [B, C]
        return logits.squeeze(-1) if logits.size(-1) == 1 else logits


@register("LSTM_SE")
def build_lstm_se(cfg, n_features, columns):
    """
    1. 說明: 依 cfg 建立 LSTM_SE
    2. inputs:
       - cfg: dict（需含 cfg["model"]）
       - n_features: int，特徵數
       - columns: list[str]，特徵名（未使用但保留以對齊介面）
    3. return:
       - nn.Module: LSTM_SE
    """
    m = cfg["model"]
    return LSTM_SE(
        input_dim=n_features,
        hidden_size=m["hidden_size"],
        n_layers=m["n_layers"],
        dropout=m["dropout"],
        bidirectional=m.get("bidirectional", False),
        num_classes=m.get("num_classes", 2),
    )


class TimeAttention(nn.Module):
    """單頭線性打分：score_t = w_a^T h_t；回傳 context 與注意力權重 alpha"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.wa = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H: torch.Tensor):
        """
        1. 說明: 時序注意力；線性打分 + softmax
        2. inputs:
           - H: [B, T, H]
        3. return:
           - context: [B, H]
           - alpha:   [B, T]
        """
        scores = self.wa(H).squeeze(-1)   # [B, T]
        alpha  = scores.softmax(dim=1)    # [B, T]
        context = (alpha.unsqueeze(-1) * H).sum(dim=1)  # [B, H]
        return context, alpha


class LSTM_Temporal(nn.Module):
    """
    LSTM → Attention → (可選 SE) → (可選 mean/max Pooling 融合) → FC → logits
    """
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        num_classes: int = 2,
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
        self.num_classes = num_classes

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

        self.attn = TimeAttention(H)

        if use_se:
            se_hidden = max(1, H // se_ratio)
            self.se = nn.Sequential(
                nn.Linear(H, se_hidden),
                nn.ReLU(),
                nn.Linear(se_hidden, H),
                nn.Sigmoid(),
            )

        head_in = H if pooling == "none" else (H * 3)
        self.ln_post = nn.LayerNorm(head_in)
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(head_in, H)
        self.fc = nn.Linear(H, num_classes)

    def forward(self, x: torch.Tensor):
        """
        1. 說明: 前向傳播；可選回傳注意力 alpha
        2. inputs:
           - x: [B, T, F]
        3. return:
           - logits: [B, C]（C=1 時 squeeze 成 [B]）
           - alpha:  [B, T]（當 return_alpha=True）
        """
        x = self.ln_in(x)
        Y, _ = self.lstm(x)                       # [B, T, H]

        context, alpha = self.attn(Y)              # [B, H], [B, T]

        if self.pooling == "meanmax":
            mean_pool = Y.mean(dim=1)             # [B, H]
            max_pool  = Y.amax(dim=1)             # [B, H]

        if self.use_se:
            gate_in = 0.5 * (mean_pool + max_pool) if self.pooling == "meanmax" else context
            w = self.se(gate_in)
            context = context * w
            if self.pooling == "meanmax":
                mean_pool = mean_pool * w
                max_pool  = max_pool  * w

        if self.pooling == "meanmax":
            feat = torch.cat([context, mean_pool, max_pool], dim=1)   # [B, 3H]
        else:
            feat = context                                            # [B, H]

        feat = self.ln_post(feat)
        feat = self.dropout(F.relu(self.hidden(feat)))                 # [B, H]
        logits = self.fc(feat)                                         # [B, C]
        logits = logits.squeeze(-1) if logits.size(-1) == 1 else logits

        if self.return_alpha:
            return logits, alpha
        return logits


@register("LSTM_Temporal")
def build_lstm_temporal(cfg, n_features, columns):
    """
    1. 說明: 依 cfg 建立 LSTM_Temporal
    2. inputs:
       - cfg: dict
       - n_features: int
       - columns: list[str]（未使用）
    3. return:
       - nn.Module: LSTM_Temporal
    """
    m = cfg["model"]
    return LSTM_Temporal(
        input_dim     = n_features,
        hidden_size   = m["hidden_size"],
        n_layers      = m["n_layers"],
        dropout       = m["dropout"],
        bidirectional = m.get("bidirectional", False),
        num_classes   = m.get("num_classes", 2),
        use_se        = m.get("use_se", False),
        se_ratio      = m.get("se_ratio", 16),
        pooling       = m.get("pooling", "none"),
        return_alpha  = m.get("return_alpha", False),
    )
