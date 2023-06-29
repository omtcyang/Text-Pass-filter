import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from ..utils import Conv_BN_ReLU


class FPNSolo1(nn.Module):
    def __init__(self, in_channels, reduce_channels, out_channels):
        super(FPNSolo1, self).__init__()

        self.reduce_layer32 = Conv_BN_ReLU(in_channels[3], reduce_channels)
        self.reduce_layer16 = Conv_BN_ReLU(in_channels[2], reduce_channels)
        self.reduce_layer8 = Conv_BN_ReLU(in_channels[1], reduce_channels)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[0], reduce_channels)

        self.up32 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up16 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up8 = nn.Upsample(scale_factor=2, mode='nearest')

        self.out32 = nn.Sequential(
            nn.Conv2d(reduce_channels, out_channels, 3, padding=1),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out16 = nn.Sequential(
            nn.Conv2d(reduce_channels, out_channels, 3, padding=1),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out8 = nn.Sequential(
            nn.Conv2d(reduce_channels, out_channels, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(reduce_channels, out_channels, 3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f4, f8, f16, f32):
        # reduce channel
        f32 = self.reduce_layer32(f32)
        f16 = self.reduce_layer16(f16)
        f8 = self.reduce_layer8(f8)
        f4 = self.reduce_layer4(f4)

        f16 = self.up32(f32) + f16  # 1/16
        f8 = self.up16(f16) + f8  # 1/8
        f4 = self.up8(f8) + f4  # 1/8

        f32 = self.out32(f32)
        f16 = self.out16(f16)
        f8 = self.out8(f8)
        f4 = self.out4(f4)

        fuse = torch.cat((f32, f16, f8, f4), 1)
        return fuse


class FPNSolo4(nn.Module):
    def __init__(self, in_channels, reduce_channels, out_channels):
        super(FPNSolo4, self).__init__()

        self.reduce_layer32 = Conv_BN_ReLU(in_channels[3], reduce_channels)
        self.reduce_layer16 = Conv_BN_ReLU(in_channels[2], reduce_channels)
        self.reduce_layer8 = Conv_BN_ReLU(in_channels[1], reduce_channels)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[0], reduce_channels)

        self.up32 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up16 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up8 = nn.Upsample(scale_factor=2, mode='nearest')

        self.out32 = nn.Conv2d(reduce_channels, out_channels, 3, padding=1, bias=False)
        self.out16 = nn.Conv2d(reduce_channels, out_channels, 3, padding=1, bias=False)
        self.out8 = nn.Conv2d(reduce_channels, out_channels, 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(reduce_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, f4, f8, f16, f32):
        # reduce channel
        f32 = self.reduce_layer32(f32)
        f16 = self.reduce_layer16(f16)
        f8 = self.reduce_layer8(f8)
        f4 = self.reduce_layer4(f4)

        f16 = self.up32(f32) + f16  # 1/16
        f8 = self.up16(f16) + f8  # 1/8
        f4 = self.up8(f8) + f4  # 1/8

        f32 = self.out32(f32)
        f16 = self.out16(f16)
        f8 = self.out8(f8)
        f4 = self.out4(f4)
        return [f32, f16, f8, f4]


class FPNSolo2(nn.Module):
    def __init__(self, in_channels, reduce_channels, out_channels):
        super(FPNSolo2, self).__init__()

        self.reduce_layer32 = Conv_BN_ReLU(in_channels[3], reduce_channels)
        self.reduce_layer16 = Conv_BN_ReLU(in_channels[2], reduce_channels)
        self.reduce_layer8 = Conv_BN_ReLU(in_channels[1], reduce_channels)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[0], reduce_channels)

        self.up32 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up16 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up8 = nn.Upsample(scale_factor=2, mode='nearest')

        self.out32 = nn.Sequential(
            nn.Conv2d(reduce_channels, out_channels, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out16 = nn.Conv2d(reduce_channels, out_channels, 3, padding=1, bias=False)
        self.out8 = nn.Sequential(
            nn.Conv2d(reduce_channels, out_channels, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out4 = nn.Conv2d(reduce_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, f4, f8, f16, f32):
        # reduce channel
        f32 = self.reduce_layer32(f32)
        f16 = self.reduce_layer16(f16)
        f8 = self.reduce_layer8(f8)
        f4 = self.reduce_layer4(f4)

        f16 = self.up32(f32) + f16  # 1/16
        f8 = self.up16(f16) + f8  # 1/8
        f4 = self.up8(f8) + f4  # 1/8

        f32 = self.out32(f32)
        f16 = self.out16(f16)
        f8 = self.out8(f8)
        f4 = self.out4(f4)
        return [torch.cat((f32, f16), 1), torch.cat((f8, f4), 1)]
