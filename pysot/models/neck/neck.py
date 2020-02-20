# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True),
            )
        self.center_size = center_size
        # self.pool = nn.Sequential(
        #         nn.Conv2d(hiddens, out_channels, kernel_size=1, bias=False),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d(kernel_size=center_size, stride=1)
        #     )

    def forward(self, x, grad=None):
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, dim=1)

        x = self.downsample(x)

        if grad is not None:
            x += grad

        if x.size(3) < 20:
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]

        # x = self.pool(x)
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0],
                                          out_channels[0],
                                          center_size)
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i],
                                            out_channels[i],
                                            center_size))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out
