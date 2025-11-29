
import torch
import torch.nn as nn
import torch.nn.functional as F

from our_model.BasicFunc import SepConv

### 小波变化
class WaveletEnhanceBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer("haar_kernel", kernel)

        self.fuse = nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False)
        self.post = SepConv(channels, channels, kernel_size=3, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        dwt = F.conv2d(x, self.haar_kernel, stride=2, groups=C)

        fea = self.fuse(dwt)  # -> [B, C, H//2, W//2]
        fea = self.post(fea)  # -> [B, C, H//2, W//2]

        out = F.interpolate(fea, size=(H, W), mode="bilinear", align_corners=False)

        return out