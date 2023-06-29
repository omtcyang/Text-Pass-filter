import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..utils import Conv_BN_ReLU


class FPN(nn.Module):
    def __init__(self, in_channels, reduce_channels, out_channels):
        super(FPN, self).__init__()
        
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], reduce_channels)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], reduce_channels)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], reduce_channels)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], reduce_channels)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.out5 = nn.Sequential(
            nn.Conv2d(reduce_channels, out_channels, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(reduce_channels, out_channels, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(reduce_channels, out_channels, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            reduce_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, f1, f2, f3, f4):
        
        # reduce channel
        f1 = self.reduce_layer1(f1)
        f2 = self.reduce_layer2(f2)
        f3 = self.reduce_layer3(f3)
        f4 = self.reduce_layer4(f4)

        f3 = self.up5(f4) + f3  # 1/16
        f2 = self.up4(f3) + f2  # 1/8
        f1 = self.up3(f2) + f1  # 1/4

        f4 = self.out5(f4)
        f3 = self.out4(f3)
        f2 = self.out3(f2)
        f1 = self.out2(f1)

        fuse = torch.cat((f4, f3, f2, f1), 1)
        return fuse
