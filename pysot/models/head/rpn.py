# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import roi_align

from pysot.models.head.xcorr import xcorr_fast, xcorr_depthwise
from pysot.models.init_weight import init_weights
from pysot.models.head.transformer import Transformer


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelRPN(RPN):
    def __init__(self, anchor_num=1, feature_in=256):
        super(UPChannelRPN, self).__init__()

        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=False),
                # nn.MaxPool2d(pool_size, stride=1),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=False),
                # nn.MaxPool2d(pool_size, stride=1),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)

        feature = xcorr_depthwise(search, kernel)
        # feature = search * kernel.expand_as(search)

        out = self.head(feature)
        return out


class DepthwiseXDiff(nn.Module):
    '''
    '''
    def __init__(self, in_channels, hidden, out_channels):
        #
        super(DepthwiseXDiff, self).__init__()
        #
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=False),
                # nn.MaxPool2d(pool_size, stride=1),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=False),
                # nn.MaxPool2d(pool_size, stride=1),
                )
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=5, bias=False, groups=hidden)
        self.head = nn.Sequential(
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)

        kernel = self.dwconv(kernel)
        search = self.dwconv(search)

        feature = search - kernel.expand_as(search)

        out = self.head(feature)
        return out


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


class MultiRPN(RPN):
    def __init__(self, in_channels, hiddens, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn'+str(i+2),
                    DepthwiseRPN(in_channels[i], hiddens[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.ctr_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        ctr = []
        # cls_head = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn'+str(idx))
            c, l, t = rpn(z_f, x_f)
            cls.append(c)
            loc.append(l)
            ctr.append(t)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)
            ctr_weight = F.softmax(self.ctr_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), \
                   weighted_avg(loc, loc_weight), \
                   weighted_avg(ctr, ctr_weight)
        else:
            return avg(cls), avg(loc), avg(ctr)
