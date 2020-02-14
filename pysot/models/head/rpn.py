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
    def __init__(self, in_channels=1024, hiddens=[512, 256]):
        super(DepthwiseRPN, self).__init__()

        self.w_cls = torch.tensor([in_channels, in_channels],
                                  dtype=torch.FloatTensor,
                                  requires_grad=True).cuda()
        self.w_loc = torch.tensor([in_channels, in_channels],
                                  dtype=torch.FloatTensor,
                                  requires_grad=True).cuda()
        nn.init.xavier_uniform(self.w_cls)
        nn.init.xavier_uniform(self.w_loc)

        self.bn_proj = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
                )

        self.head_cls = self._head(in_channels, hiddens)
        self.head_loc = self._head(in_channels, hiddens)

        self.cls = nn.Conv2d(hiddens, 2, kernel_size=1)
        self.loc = nn.Conv2d(hiddens, 4, kernel_size=1)
        self.ctr = nn.Conv2d(hiddens, 1, kernel_size=1)

    def _proj(self, z):
        '''
        z_f: (N, nch, 1, 1)
        '''
        nb, nch = z_f.shape[:2]
        w_cls = self.w_cls.repeat(nb, 1) # (N*nch, nch)
        w_loc = self.w_loc.repeat(nb, 1) # (N*nch, nch)

        zf = z.view(-1, 1) # (N*nch, 1)
        w_cls = w_cls * zf.expand_as(w_cls)
        w_loc = w_loc * zf.expand_as(w_loc)

        return w_cls, w_loc

    def _head(self, in_channels, hiddens):
        '''
        '''
        layers = []
        for ich, och in zip([in_channels]+hiddens[:-1], hiddens):
            layers.append(nn.Conv2d(ich, och, 1, bias=False))
            layers.append(nn.BatchNorm2d(och))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, z_f, x_f):
        if isinstance(z_f, (list, tuple)):
            z_f = torch.cat(z_f, dim=1)
            x_f = torch.cat(x_f, dim=1)

        kernel_cls, kernel_loc = self._proj(z_f)

        feat_cls = xcorr_proj(x_f, kernel_cls)
        feat_loc = xcorr_proj(x_f, kernel_loc)

        feat_cls = self.head_cls(feat_cls)
        feat_loc = self.head_loc(feat_loc)

        cls = self.cls(feat_cls)
        ctr = self.cls(feat_ctr)
        loc = self.loc(feat_loc)
        return cls, loc, ctr

