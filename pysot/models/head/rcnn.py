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

    def forward(self, z_f, x_f):
        raise NotImplementedError


class DepthwiseRCNN(RCNN):
    def __init__(self, in_channels=256, in_channels_z=576, preproc_channels=256, hiddens=256):
        super(DepthwiseRCNN, self).__init__()
        self.preproc_z = nn.Sequential(
                nn.Conv2d(in_channels_z, in_channels_z, 3, bias=False, groups=in_channels_z),
                nn.BatchNorm2d(in_channels_z),
                nn.Conv2d(in_channels_z, preproc_channels, 1, bias=False),
                nn.BatchNorm2d(preproc_channels),
                nn.ReLU(inplace=True)
                )
        self.preproc_x = nn.Sequential(
                nn.Conv2d(in_channels, preproc_channels, 1, bias=False),
                nn.BatchNorm2d(preproc_channels),
                nn.ReLU(inplace=True)
                )
        self.head = nn.Sequential(
                # fc layers with bottleneck
                nn.Conv2d(preproc_channels, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 768, kernel_size=1, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True),
                nn.Conv2d(768, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 768, kernel_size=1, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True),
                # nn.Conv2d(preproc_channels, hiddens, kernel_size=1, bias=False),
                # nn.BatchNorm2d(hiddens),
                # nn.ReLU(inplace=True),
                )
        self.ctr = nn.Sequential(
                nn.Conv2d(768, hiddens, kernel_size=1, bias=False),
                nn.BatchNorm2d(hiddens),
                nn.ReLU(inplace=True),
                nn.Conv2d(hiddens, 1, kernel_size=1)
                )
        self.cls = nn.Sequential(
                nn.Conv2d(768, hiddens, kernel_size=1, bias=False),
                nn.BatchNorm2d(hiddens),
                nn.ReLU(inplace=True),
                nn.Conv2d(hiddens, 2, kernel_size=1)
                )
        self.loc = nn.Sequential(
                nn.Conv2d(768, hiddens, kernel_size=1, bias=False),
                nn.BatchNorm2d(hiddens),
                nn.ReLU(inplace=True),
                nn.Conv2d(hiddens, 4, kernel_size=1)
                )

    def forward(self, z_f, x_f):
        if isinstance(z_f, (list, tuple)):
            z_f = torch.cat(z_f, dim=1)
            x_f = torch.cat(x_f, dim=1)

        search = self.preproc_x(x_f)
        kernel = self.preproc_z(z_f)
        feature = xcorr_depthwise(search, kernel)
        feature = self.head(feature)

        ctr = self.ctr(feature)
        cls = self.cls(feature)
        loc = self.loc(feature)
        return ctr, cls, loc
