# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import roi_align

from pysot.models.head.xcorr import xcorr_fast, xcorr_depthwise #, L1DiffFunction
from pysot.models.init_weight import init_weights


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class DepthwiseRPN(RPN):
    def __init__(self, in_channels=256, hiddens=256):
        super(DepthwiseRPN, self).__init__()
        self.preproc = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, bias=False, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
                )
        self.head = nn.Sequential(
                nn.Conv2d(in_channels, hiddens, kernel_size=1, bias=False),
                nn.BatchNorm2d(hiddens),
                nn.ReLU(inplace=True)
                )
        self.cls = nn.Sequential(
                nn.Conv2d(hiddens, hiddens, kernel_size=1, bias=False),
                nn.BatchNorm2d(hiddens),
                nn.ReLU(inplace=True),
                nn.Conv2d(hiddens, 2, kernel_size=1)
                )
        self.loc = nn.Sequential(
                nn.Conv2d(hiddens, hiddens, kernel_size=1, bias=False),
                nn.BatchNorm2d(hiddens),
                nn.ReLU(inplace=True),
                nn.Conv2d(hiddens, 4, kernel_size=1)
                )
        self.ctr = nn.Sequential(
                nn.Conv2d(hiddens, hiddens, kernel_size=1, bias=False),
                nn.BatchNorm2d(hiddens),
                nn.ReLU(inplace=True),
                nn.Conv2d(hiddens, 1, kernel_size=1)
                )

    def forward(self, z_f, x_f):
        if isinstance(z_f, (list, tuple)):
            z_f = torch.cat(z_f, dim=1)
            x_f = torch.cat(x_f, dim=1)

        search = self.preproc(x_f)
        kernel = self.preproc(z_f)
        feature = self.head(xcorr_depthwise(search, kernel))

        cls = self.cls(feature)
        loc = self.loc(feature)
        ctr = self.ctr(feature)
        return cls, loc, ctr

