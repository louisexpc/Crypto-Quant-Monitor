import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveKernelCNN(nn.Module):
    """
    多 kernel 1D 卷積 + 注意力加權 + 殘差，時間長度 T 可變。
    x: [B, C, T]（例如 C=OHLC 或你擴充後的快特徵）
    """
    def __init__(self, in_channels=4, hidden=64, out_dim=64, kernel_sizes=(3,7,15), dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, hidden, k, padding=k//2),
                nn.GELU(),
                nn.GroupNorm(num_groups=8, num_channels=hidden)  # 小 batch 友善
            ) for k in kernel_sizes
        ])
        # 對每個 kernel 的 pooled 向量各自打分：Linear(64→1)
        self.score = nn.Linear(hidden, 1, bias=False)

        # 殘差：時間不變設計（1x1 conv 對 channel 做投影，再全域池化）
        self.residual = nn.Conv1d(in_channels, hidden, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden, out_dim)

    def forward(self, x):                       # x: [B, C, T]
        feats = [br(x) for br in self.branches] # list of [B, H, T]
        pooled = [F.adaptive_avg_pool1d(f, 1).squeeze(-1) for f in feats]  # list of [B, H]
        P = torch.stack(pooled, dim=1)          # [B, K, H]

        scores = self.score(P)                  # [B, K, 1]
        alpha = F.softmax(scores, dim=1)        # [B, K, 1]
        F_stack = torch.stack(feats, dim=1)     # [B, K, H, T]
        fused = (F_stack * alpha.unsqueeze(-1)).sum(dim=1)  # [B, H, T]

        # 時間不變殘差
        res = self.residual(x)                  # [B, H, T]
        fused = fused + res

        # 池化 → 投影
        g = F.adaptive_avg_pool1d(fused, 1).squeeze(-1)     # [B, H]
        g = self.dropout(g)
        return self.proj_out(g)                              # [B, out_dim]