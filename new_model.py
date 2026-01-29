
import torch
import torch.nn as nn

from our_model.BasicFunc import Downsample
from our_model.BasicFunc import Upsample
from our_model.GrayWorldRetinex import GrayWorldRetinex
from our_model.BasicLayer import BasicLayer
from our_model.EfficientViMBlock import EfficientViMBlock



class NewModel(nn.Module):
    def __init__(self,in_channels=3,feature_channels=32,use_white_balance=False):
        super(NewModel,self).__init__()
        self.use_white_balance = use_white_balance

        if self.use_white_balance:
            self.wb = GrayWorldRetinex()
            self.alpha = nn.Parameter(torch.zeros(1,3,1,1))

        self.first = nn.Conv2d(
            in_channels, feature_channels, kernel_size=3, stride=1, padding=1
        )

        self.encoder1 = BasicLayer(feature_channels)
        self.down1 = Downsample(feature_channels)

        self.encoder2 = BasicLayer(feature_channels * 2 ** 1)
        self.down2 = Downsample(feature_channels * 2 ** 1)

        # self.encoder3 = BasicLayer(feature_channels * 2 ** 2)
        # self.down3 = Downsample(feature_channels * 2 ** 2)

        self.bottleneck = BasicLayer(feature_channels * 2 ** 2)  # 瓶颈层 128通道
        # self.bottleneck = BasicLayer(feature_channels * 2 ** 3)  # 瓶颈层 256通道

        # self.up0 = Upsample(feature_channels * 2 ** 3)
        # self.decoder0 = BasicLayer(feature_channels * 2 ** 2) #128

        self.up1 = Upsample(feature_channels * 2 ** 2)
        self.decoder1 = BasicLayer(feature_channels * 2 ** 1) #64

        self.up2 = Upsample(feature_channels * 2 ** 1)
        self.decoder2 = BasicLayer(feature_channels) #32

        # 转回3通道 RGB输出
        self.out = nn.Conv2d(
            feature_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self,x):
        res = x

        #####2 -  1 - 2
        if self.use_white_balance:
            alpha = torch.sigmoid(self.alpha)
            x = alpha * self.wb(x) + (1 - alpha) * x
        x1 = self.first(x)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))
        x = self.up1(x3) + x2
        x = self.decoder1(x)
        x = self.up2(x) + x1
        x = self.decoder2(x)
        out = self.out(x) + res
        return out

        ##### 1-1-1
        # if self.use_white_balance:
        #     alpha = torch.sigmoid(self.alpha)
        #     x = alpha * self.wb(x) + (1 - alpha) * x
        #
        # # Encoder
        # x1 = self.encoder1(self.first(x)) #32
        # # Bottleneck
        # x2 = self.bottleneck(self.down1(x1)) #64
        #
        # # Decoder (single stage)
        # x = self.decoder2(self.up2(x2) + x1)
        # # Output with global residual
        # out = self.out(x) + res
        # return out

        ###### 3 - 1 - 3
        # if self.use_white_balance:
        #     alpha = torch.sigmoid(self.alpha)
        #     x = alpha * self.wb(x) + (1 - alpha) * x
        # x1 = self.first(x)
        # x1 = self.encoder1(x1)
        # x2 = self.encoder2(self.down1(x1)) # 64
        # x3 = self.encoder3(self.down2(x2))
        # x4 = self.bottleneck(self.down3(x3)) #256
        # x = self.up0(x4) + x3 # 128
        # x = self.decoder0(x)
        # x = self.up1(x) + x2 # 64
        # x = self.decoder1(x)
        # x = self.up2(x) + x1 #32
        # x = self.decoder2(x)
        # out = self.out(x) + res
        # return out