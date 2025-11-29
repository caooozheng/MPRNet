
import torch
import torch.nn as nn
import torch.nn.functional as F

from our_model.BasicFunc import SepConv

## HIN Block
class BasicBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, relu_slope=0.1):
        super(BasicBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = SepConv(in_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_2 = SepConv(out_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=True)
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)

    def forward(self, x):
        out = self.conv_1(x)

        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out = out + self.identity(x)
        return out


class GetGradient(nn.Module):
    def __init__(self, dim=3, mode="sobel"):
        super(GetGradient, self).__init__()
        self.dim = dim
        self.mode = mode
        if mode == "sobel":
            # sobel filter
            kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

            kernel_y = (
                torch.tensor(kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            kernel_x = (
                torch.tensor(kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )

            self.register_buffer("kernel_y", kernel_y.repeat(self.dim, 1, 1, 1))
            self.register_buffer("kernel_x", kernel_x.repeat(self.dim, 1, 1, 1))
        elif mode == "laplacian":
            kernel_laplace = [[0.25, 1, 0.25], [1, -5, 1], [0.25, 1, 0.25]]
            kernel_laplace = (
                torch.tensor(kernel_laplace, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            self.register_buffer(
                "kernel_laplace", kernel_laplace.repeat(self.dim, 1, 1, 1)
            )

    def forward(self, x):
        if self.mode == "sobel":
            grad_x = F.conv2d(x, self.kernel_x, padding=1, groups=self.dim)
            grad_y = F.conv2d(x, self.kernel_y, padding=1, groups=self.dim)

            grad_magnitude = torch.sqrt(
                torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 1e-6
            )
        elif self.mode == "laplacian":
            grad_magnitude = F.conv2d(
                x, self.kernel_laplace, padding=1, groups=self.dim
            )
            grad_magnitude = torch.abs(grad_magnitude)  # magnitude only

        return grad_magnitude





class SGFB(nn.Module):
    def __init__(self, feature_channels=48):
        super(SGFB, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.frdb1 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.frdb2 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.get_gradient = GetGradient(feature_channels, mode="sobel")
        self.conv_grad = nn.Sequential(
            SepConv(feature_channels, feature_channels, kernel_size=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        grad = self.get_gradient(x)
        grad = self.conv_grad(grad)
        x = self.frdb1(x)
        alpha = torch.sigmoid(self.alpha)
        x = alpha * grad * x + (1 - alpha) * x
        x = self.frdb2(x)
        return x
