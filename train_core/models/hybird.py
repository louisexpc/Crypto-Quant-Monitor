import torch
import torch.nn as nn
import torch.nn.functional as F

from fastCNN import AdaptiveKernelCNN
from Transformer import TemporalTransformer

class BilinearFusion(nn.Module):
    """
    雙線性交叉融合模組，用來融合 CNN 與 Transformer 的特徵。
    """
    def __init__(self, cnn_dim: int, trans_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_dim, cnn_dim, trans_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, cnn_feat, trans_feat):
        """
        cnn_feat: [B, cnn_dim]
        trans_feat: [B, trans_dim]
        回傳: [B, out_dim]
        """
        # 雙線性交叉融合：每一個輸出維度都是 cnn_feat^T * W_i * trans_feat
        fusion = torch.einsum('bi,oij,bj->bo', cnn_feat, self.weight, trans_feat) + self.bias
        return fusion

class FactorizedBilinear(nn.Module):
    """
    MLB: 以 rank-r 做低秩雙線性融合，顯著降參數/過擬合風險。
    """
    def __init__(self, cnn_dim: int, trans_dim: int, out_dim: int, rank: int = 16, dropout=0.1):
        super().__init__()
        self.fc_c = nn.Linear(cnn_dim,  out_dim * rank, bias=False)
        self.fc_t = nn.Linear(trans_dim, out_dim * rank, bias=False)
        self.bias  = nn.Parameter(torch.zeros(out_dim))
        self.drop  = nn.Dropout(dropout)

    def forward(self, c, t):                   # c: [B, C], t: [B, Tdim]
        C = self.fc_c(c).view(c.size(0), -1, 1)              # [B, out*r, 1]
        T = self.fc_t(t).view(t.size(0), -1, 1)              # [B, out*r, 1]
        z = (C.view(c.size(0), -1) * T.view(t.size(0), -1))  # [B, out*r]
        z = z.view(c.size(0), -1, 16).sum(-1)                # [B, out]，r=16
        # 穩定化（可選）：signed-sqrt + L2-norm
        z = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-8)
        z = F.normalize(z, p=2, dim=-1)
        return self.drop(z + self.bias)


class CNNTransformerBilinearModel(nn.Module):
    def __init__(self,
                 cnn_in_channels=4,
                 cnn_out_dim=64,
                 trans_input_dim=16,
                 trans_d_model=128,
                 trans_n_heads=4,
                 trans_layers=3,
                 fuse_out_dim=128,
                 num_classes=2,                     # up/flat/down
                 rank=16,
                 dropout=0.2):
        super().__init__()
        self.cnn = AdaptiveKernelCNN(in_channels=cnn_in_channels,
                                     hidden=64, out_dim=cnn_out_dim, dropout=dropout)

        # 你的 TemporalTransformer 務必回 [B, trans_d_model]
        self.trans = TemporalTransformer(input_dim=trans_input_dim,
                                         d_model=trans_d_model,
                                         num_classes=0,   # 請確保回 pooled embedding
                                         n_heads=trans_n_heads,
                                         num_layers=trans_layers)

        self.fuse = FactorizedBilinear(cnn_dim=cnn_out_dim,
                                       trans_dim=trans_d_model,
                                       out_dim=fuse_out_dim,
                                       rank=rank, dropout=dropout)

        # 主分類頭
        self.head = nn.Sequential(
            nn.LayerNorm(fuse_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_out_dim, num_classes)
        )
        # 兩個輔助頭（deep supervision，前期更穩）
        self.aux_cnn   = nn.Linear(cnn_out_dim, num_classes)
        self.aux_trans = nn.Linear(trans_d_model, num_classes)

    def forward(self, x_fast, x_slow):
        """
        x_fast: [B, C_fast, T_fast]   # 15m 快窗（例：最近 8~16 根）
        x_slow: [B, T_slow, F_slow]   # 1H 慢窗（例：最近 96~168 根）
        """
        f_c = self.cnn(x_fast)        # [B, D_c]
        f_t = self.trans(x_slow)      # [B, D_t]
        z   = self.fuse(f_c, f_t)     # [B, D_z]
        logits = self.head(z)         # [B, num_classes]
        # 同時回輔助頭以便訓練期加權
        return logits, self.aux_cnn(f_c), self.aux_trans(f_t)