# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import roi_align

from pysot.models.head.xcorr import xcorr_depthwise #, L1DiffFunction
from pysot.models.init_weight import init_weights


class RCNN(nn.Module):
    def __init__(self):
        super(RCNN, self).__init__()

    def forward(self, feat):
        raise NotImplementedError


class DepthwiseRCNN(RCNN):
    def __init__(self, in_channels=256, channels=[256, 128, 128], hiddens=512):
        super(DepthwiseRCNN, self).__init__()

        self.preproc = self._dwconv3(in_channels, hiddens)
        self.ctr = nn.Sequential(
                self._dwconv3(hiddens, channels[0]),
                nn.Conv2d(channels[0], 1, kernel_size=1)
                )
        self.cls = nn.Sequential(
                self._dwconv3(hiddens, channels[1]),
                nn.Conv2d(channels[1], 2, kernel_size=1)
                )
        self.loc = nn.Sequential(
                self._dwconv3(hiddens, channels[2]),
                nn.Conv2d(channels[2], 4, kernel_size=1)
                )

    def _dwconv3(self, ich, och):
        return nn.Sequential(
                nn.Conv2d(ich, ich, 3, stride=1, bias=False, groups=ich),
                nn.BatchNorm2d(ich),
                nn.Conv2d(ich, och, 1, bias=False),
                nn.BatchNorm2d(och),
                nn.ReLU(inplace=True)
                )

    def forward(self, feat):
        if isinstance(feat, (list, tuple)):
            feat = torch.cat(feat, dim=1)

        feat = self.preproc(feat)

        ctr = self.ctr(feat)
        cls = self.cls(feat)
        loc = self.loc(feat)
        return ctr, cls, loc

