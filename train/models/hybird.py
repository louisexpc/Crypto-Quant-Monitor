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


class CNNTransformerBilinearModel(nn.Module):
    def __init__(self,
                 cnn_in_channels=4,
                 cnn_out_dim=64,
                 trans_input_dim=16,
                 trans_d_model=128,
                 trans_n_heads=4,
                 trans_layers=3,
                 bilinear_out_dim=128,
                 num_classes=2):
        super().__init__()
        # CNN 模組
        self.cnn = AdaptiveKernelCNN(in_channels=cnn_in_channels,
                                     out_dim=cnn_out_dim)
        
        # Transformer 模組（已去除輸出）
        self.trans = TemporalTransformer(input_dim=trans_input_dim,
                                         d_model=trans_d_model,
                                         num_classes=0,  # 不要 head
                                         n_heads=trans_n_heads,
                                         num_layers=trans_layers)
        
        # 融合
        self.bilinear = BilinearFusion(cnn_dim=cnn_out_dim,
                                       trans_dim=trans_d_model,
                                       out_dim=bilinear_out_dim)
        
        # 最終分類頭
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(bilinear_out_dim, num_classes)
        )

    def forward(self, cnn_x, trans_x):
        """
        cnn_x: [B, C, T] → 適合短時間高頻特徵（如 15m K 線）
        trans_x: [B, T, F] → 適合長期時序（如 1H K 線）
        """
        cnn_feat = self.cnn(cnn_x)            # [B, 64]
        trans_feat = self.trans(trans_x)      # [B, 128]
        fused = self.bilinear(cnn_feat, trans_feat)  # [B, 128]
        logits = self.head(fused)             # [B, num_classes]
        return logits
