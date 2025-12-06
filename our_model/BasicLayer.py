import torch
import torch.nn as nn

from our_model.WaveletEnhanceBlock import WaveletEnhanceBlock
from our_model.SGFB import SGFB
from our_model.Attention import Attention


class BasicLayer(nn.Module):
    def __init__(self, feature_channels=32):
        super(BasicLayer, self).__init__()
       # self.ln = nn.LayerNorm(feature_channels,eps=1e-6,elementwise_affine=False)
        self.web = WaveletEnhanceBlock(feature_channels)

        self.attn = Attention(feature_channels)

        # self.fusion_gate = nn.Sequential(
        #     nn.Conv2d(feature_channels * 2, 2, kernel_size=1),  # 将 (out1, out2) 映射到 (weight1, weight2)
        #     nn.Sigmoid()  # 使用 Sigmoid 将权重归一化到 (0, 1)
        # )



        self.sgfb = SGFB(feature_channels)

        # 动态路由门控网络 (Routing Gate)
        # 1. 输入是拼接的 [B, 2C, H, W]
        # 2. 输出是 [B, 1, H, W]，这个 1 通道将作为选择权重 M
        self.routing_gate = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels // 2, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feature_channels // 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()  # Sigmoid 将权重 M 约束在 [0, 1] 之间
        )



    # def forward(self, x):
    #     res = x
    #     out1 = self.web(x)
    #     out2 = self.attn(x)
    #
    #     # 2. 融合：拼接 out1 和 out2，然后通过门控机制
    #     # 张量形状：[B, C, H, W] -> cat -> [B, 2C, H, W]
    #     # fusion_gate 输出形状：[B, 2, H, W]
    #     gate = self.fusion_gate(torch.cat([out1, out2], dim=1))
    #
    #     # 3. 提取权重
    #     # weight1 形状：[B, 1, H, W]
    #     # weight2 形状：[B, 1, H, W]
    #     weight1, weight2 = gate.chunk(2, dim=1)
    #
    #     # 4. 加权求和融合，得到新的特征 x_fused
    #     # element-wise multiplication (*)
    #     x_fused = weight1 * out1 + weight2 * out2
    #
    #     # 5. 输入到 sgfb
    #     x = self.sgfb(x_fused)  # 注意：这里使用融合后的 x_fused
    #
    #
    #
    #     # x = self.sgfb(x)
    #     return 0.5 * x + 0.5 * res

    def forward(self, x):
        res = x  # 最终残差连接的输入

        # 1. 并行处理 (假设您采纳了我的建议，移除了模块内部残差，以增强差异化)
        # 如果您仍保留内部残差，则使用 out1 = self.web(x) + x
        out1 = self.web(x) + x
        out2 = self.attn(x) + x

        # 2. 拼接特征并计算路由掩码 M
        # 形状：[B, 2C, H, W]
        combined_features = torch.cat([out1, out2], dim=1)

        # M 形状：[B, 1, H, W]，值域 [0, 1]
        M = self.routing_gate(combined_features)

        # 3. 动态选择/融合
        # 融合公式：x_fused = M * out1 + (1 - M) * out2
        # M 接近 1 时，选择 out1；M 接近 0 时，选择 out2
        x_fused = M * out1 + (1 - M) * out2

        # 4. 输入到 sgfb
        x_out = self.sgfb(x_fused)

        # 5. 最终残差连接
        # 0.5 * x_out 是 sgfb 处理后的特征
        # 0.5 * res 是原始输入 x (即最终的残差)
        return 0.5 * x_out + 0.5 * res
