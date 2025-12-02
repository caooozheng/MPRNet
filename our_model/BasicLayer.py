import torch
import torch.nn as nn

from our_model.WaveletEnhanceBlock import WaveletEnhanceBlock
from our_model.SGFB import SGFB
from our_model.Attention import Attention


class BasicLayer(nn.Module):
    def __init__(self, feature_channels=32):
        super(BasicLayer, self).__init__()
       # self.ln = nn.LayerNorm(feature_channels,eps=1e-6,elementwise_affine=False)
       # self.web = WaveletEnhanceBlock(feature_channels)

        self.attn = Attention(feature_channels)

        self.sgfb = SGFB(feature_channels)

    def forward(self, x):
        res = x
        #x = self.web(x) + x
        x = self.attn(x) + x
        x = self.sgfb(x)
        return 0.5 * x + 0.5 * res
