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
from pysot.models.head.transformer import Transformer


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class DepthwiseRPN(RPN):
    def __init__(self, num_anchor, in_channels=256, hiddens=256):
        super(DepthwiseRPN, self).__init__()
        self.preproc_z = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, bias=False, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
                )
        self.preproc_x = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, bias=False, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
                )
        self.head = nn.Sequential(
                nn.Conv2d(in_channels, hiddens, kernel_size=1, bias=False),
                nn.BatchNorm2d(hiddens),
                nn.ReLU(inplace=True)
                )
        self.asp = nn.Sequential(
                nn.Conv2d(hiddens, hiddens, kernel_size=1, bias=False),
                nn.BatchNorm2d(hiddens),
                nn.ReLU(inplace=True),
                nn.Conv2d(hiddens, 1, kernel_size=1, bias=True)
                )
        self.ctr = nn.Sequential(
                nn.Conv2d(hiddens, hiddens, kernel_size=1, bias=False),
                nn.BatchNorm2d(hiddens),
                nn.ReLU(inplace=True),
                nn.Conv2d(hiddens, 1, kernel_size=1, bias=True)
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
        asp = self.asp(feature)
        return ctr, asp
