# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision.ops import roi_align

from pysot.models.head.xcorr import xcorr_fast, xcorr_depthwise #, L1DiffFunction
from pysot.models.init_weight import init_weights


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class DepthwiseRPN(RPN):
    def __init__(self, in_channels=256, hiddens=256, kh=5, kw=5):
        super(DepthwiseRPN, self).__init__()
        n_preds = kh * kw
        self.preproc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, bias=False, groups=in_channels),
            nn.GroupNorm(in_channels//16, in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(in_channels//16, in_channels),
            nn.ReLU(inplace=True)
            )
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hiddens, kernel_size=1, bias=False),
            nn.GroupNorm(hiddens//16, hiddens),
            nn.ReLU(inplace=True)
            )
        self.cls_feat = nn.Sequential(
            nn.Conv2d(hiddens, hiddens, kernel_size=1, bias=False),
            nn.GroupNorm(hiddens//16, hiddens, affine=True),
            # nn.ReLU(inplace=True),
            )
        self.cls = nn.Conv2d(hiddens, 1*n_preds, kernel_size=1)
        self.loc = nn.Sequential(
            nn.Conv2d(hiddens, hiddens, kernel_size=1, bias=False),
            nn.GroupNorm(hiddens//16, hiddens),
            nn.ReLU(inplace=True),
            nn.Conv2d(hiddens, 4*n_preds, kernel_size=1)
            )
        self.ctr_feat = nn.Sequential(
            nn.Conv2d(hiddens, hiddens, kernel_size=1, bias=False),
            nn.GroupNorm(hiddens//16, hiddens),
            nn.ReLU(inplace=True)
        )
        self.ctr = nn.Conv2d(hiddens, 1*n_preds, kernel_size=1)

        self.cls_weight = nn.Parameter(torch.zeros(n_preds))
        self.ctr_weight = nn.Parameter(torch.zeros(n_preds))
        self.loc_weight = nn.Parameter(torch.zeros(4, n_preds))

        self.kh = kh
        self.kw = kw

    def _ps_rearrange(self, feat, kh, kw):
        '''
        '''
        _, nch, ww, hh = feat.shape
        ngroup = kw * kh
        splits = torch.split(feat, nch//ngroup, dim=1)
        nw = ww - (kw - 1)
        nh = hh - (kh - 1)
        k = 0
        res = []
        for y in range(kh):
            for x in range(kw):
                res.append(splits[k][:, :, y:y+nh, x:x+nw])
                k += 1
        return torch.cat(res, dim=1)

    def forward(self, z_f, x_f):
        if isinstance(z_f, (list, tuple)):
            z_f = torch.cat(z_f, dim=1)
            x_f = torch.cat(x_f, dim=1)

        search = self.preproc(x_f)
        kernel = self.preproc(z_f)
        feature = self.head(xcorr_depthwise(search, kernel))

        cls_feat = self.cls_feat(feature)
        # cls_feat = F.normalize(cls_feat, p=2, dim=1)
        cls = self.cls(cls_feat)
        cls = self._ps_rearrange(cls, self.kh, self.kw)
        # cls = self.cls(feature)
        loc = self.loc(feature)
        w2, h2 = self.kw // 2, self.kh // 2
        # loc = loc[:, :, h2:-h2, w2:-w2]
        loc = self._ps_rearrange(loc, self.kh, self.kw)
        ctr_feat = self.ctr_feat(feature)
        ctr = self.ctr(ctr_feat)
        ctr = self._ps_rearrange(ctr, self.kh, self.kw)

        cls_weight = F.softmax(self.cls_weight, dim=-1)
        ctr_weight = F.softmax(self.ctr_weight, dim=-1)
        loc_weight = F.softmax(self.loc_weight, dim=-1)
        return torch.sigmoid(cls), loc, torch.sigmoid(ctr), cls_weight, loc_weight, ctr_weight