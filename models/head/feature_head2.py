#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time : 2023/1/3 11:43
@Author : Haozhao Ma
@Email : haozhaoma@mail.nwpu.edu.cn
@time: 2023/1/3 11:43
"""
import torch
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d

import math
from ..utils import Conv_BN_ReLU


class FeatureHead2(nn.Module):
    def __init__(self):
        super(FeatureHead2, self).__init__()
        self.convs_all_levels = nn.ModuleList()
        for i in range(4):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = Conv_BN_ReLU(128, 64, 3, padding=1)
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = 130 if i == 3 else 128
                    one_conv = Conv_BN_ReLU(chn, 64, 3, padding=1)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module('upsample' + str(j), one_upsample)
                    continue

                one_conv = Conv_BN_ReLU(64, 64, 3, padding=1)
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = Conv_BN_ReLU(64, 64, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == 3:
                input_feat = input_p
                x_range = torch.linspace(-1, 1, input_feat.shape[-1], device=input_feat.device)
                y_range = torch.linspace(-1, 1, input_feat.shape[-2], device=input_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([input_feat.shape[0], 1, -1, -1])
                x = x.expand([input_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                input_p = torch.cat([input_p, coord_feat], 1)

            feature_add_all_level += self.convs_all_levels[i](input_p)

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred
