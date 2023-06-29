import torch
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d

import math
from ..utils import Conv_BN_ReLU


class FeatureHead(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(FeatureHead, self).__init__()
        self.feature_head = Conv_BN_ReLU(in_channels, hidden_dim, 3, padding=1)
        # self.ins = Conv_BN_ReLU(hidden_dim, 1, 3, padding=1)
        # self.ins = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fuse):
        feature = self.feature_head(fuse)
        # if self.training:
        #     return feature, self.ins(feature)
        return feature
