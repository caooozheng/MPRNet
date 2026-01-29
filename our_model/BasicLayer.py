import torch
import torch.nn as nn

from our_model.WaveletEnhanceBlock import WaveletEnhanceBlock
from our_model.SGFB import SGFB
from our_model.Attention import Attention
from our_model.EfficientViMBlock import EfficientViMBlock

class BasicLayer(nn.Module):
    def __init__(self, feature_channels=32):
        super(BasicLayer, self).__init__()
        # self.web = WaveletEnhanceBlock(feature_channels)

        self.attn = Attention(feature_channels)
        self.vim = EfficientViMBlock(feature_channels)
        # self.sgfb = SGFB(feature_channels)

        self.routing_gate = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels // 2, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feature_channels // 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()  # Sigmoid 将权重 M 约束在 [0, 1] 之间
        )

    # def forward(self, x):
    #     res = x  # 最终残差连接的输入
    #     out1 = self.vim(x) + x
    #     # out2 = self.attn(x) + x
    #     # combined_features = torch.cat([out1, out2], dim=1)
    #     # M = self.routing_gate(combined_features)
    #     # x_fused = M * out1 + (1 - M) * out2
    #     # x_out = self.sgfb(x_fused)
    #     return  0.5 * x_out + 0.5 * res


    def forward(self, x):
        res = x
        x = self.attn(x) + x
        x = self.vim(x)
        return 0.5 * x + 0.5 * res

