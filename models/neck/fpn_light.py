import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..utils import Conv_BN_ReLU


class Light_FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Light_FPN, self).__init__()
        planes = out_channels

        self.dwconv4_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer4_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv3_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv1_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, f1, f2, f3, f4):
        f3 = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2 = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3, f2)))
        f1 = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2, f1)))
        f4 = self.smooth_layer4_1(self.dwconv4_1(f4))

        return f1, f2, f3, f4
