
import torch
import torch.nn as nn

class GrayWorldRetinex(nn.Module):
    def __init__(self, eps=1e-6):
        super(GrayWorldRetinex, self).__init__()
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        mean = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        gray_mean = mean.mean(dim=1, keepdim=True)  # [B, 1, 1, 1]
        gain = gray_mean / (mean + self.eps)
        x = x * gain  # white balance
        x_log = torch.log(x + self.eps)
        x_log = x_log - x_log.mean(dim=(2, 3), keepdim=True)
        x_out = torch.exp(x_log)
        x_min = x_out.amin(dim=(-2, -1), keepdim=True)
        x_max = x_out.amax(dim=(-2, -1), keepdim=True)
        x_out = (x_out - x_min) / (x_max - x_min + self.eps)
        return x_out
