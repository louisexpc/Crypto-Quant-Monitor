import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveKernelCNN(nn.Module):
    """
    CNN 模組，根據市場波動度動態加權多種 kernel size，融合不同感受野特徵並加入殘差。
    """
    def __init__(self, in_channels=4, out_dim=64, kernel_sizes=(3, 7, 15)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, 64, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.att_fc = nn.Linear(64, len(kernel_sizes))  # 從每個 conv 輸出的 pooled 向量計算權重
        self.residual_proj = nn.Linear(in_channels * 4, 64)  # 原始輸入殘差投影
        self.out = nn.Linear(64, out_dim)

    def forward(self, x):
        """
        x: [B, C, T]，例如 [B, 4, 4]
        """
        conv_outputs = [conv(x) for conv in self.convs]        # List of [B, 64, T]
        # 對每個 kernel 的特徵輸出做 pooling 得到 [B, 64] 向量
        pooled_vectors = [F.adaptive_avg_pool1d(c, 1).squeeze(-1) for c in conv_outputs]  # List of [B, 64]
        pooled_stack = torch.stack(pooled_vectors, dim=1)      # [B, K, 64]

        # 透過 Linear 計算權重 再 softmax 權重
        alpha = F.softmax(self.att_fc(pooled_stack.mean(dim=1)), dim=-1)  # [B, K]
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)              # [B, K, 1, 1]

        # 原始 conv 輸出堆疊
        stacked = torch.stack(conv_outputs, dim=1)            # [B, K, 64, T]
        fused = (stacked * alpha).sum(dim=1)                  # [B, 64, T]

        # 池化 + 殘差
        pooled_fused = F.adaptive_avg_pool1d(fused, 1).squeeze(-1)  # [B, 64]
        residual = self.residual_proj(x.flatten(start_dim=1))       # [B, 64]

        out = self.out(pooled_fused + residual)                     # [B, out_dim]
        return out

dummy_input = torch.randn(8, 4, 4)  # 8 筆樣本、4 個特徵（OHLC）、4 根 15mK線
model = AdaptiveKernelCNN(in_channels=4, out_dim=64)
output = model(dummy_input)        # output.shape: [8, 64]
