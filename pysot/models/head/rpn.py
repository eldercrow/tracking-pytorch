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
    def __init__(self, anchor_num=5, feature_in=256):
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
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=False),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=False),
                )
        self.transformer = Transformer(25*2, [64, 128, 256, 512], 25*25)
        # if use_transformer:
        #     self.transformer = Transformer(25*2, [128, 256, 512, 1024], 25*25)
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )

    def forward(self, kernel, search, search_crop=None, search_pos=None):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        if search_crop is not None:
            search_crop = self.conv_search(search_crop)

        if search_pos is not None:
            search_pos += 1.5
            search_pos = [s for s in search_pos]
            search_crop = roi_align(search, search_pos, output_size=(5, 5))
        if search_crop is not None:
            kernel_t, w = self.transformer(kernel, search_crop)
            kernel_t = kernel * w[:, 0:1, :, :] + kernel_t * w[:, 1:, :, :]
        else:
            kernel_t = kernel

        feature = xcorr_depthwise(search, kernel_t)
        out = self.head(feature)
        return out


class DepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        # self.cls = PSDWDiff(in_channels, ps_channels, out_channels, 2 * anchor_num)
        # self.loc = PSDWDiff(in_channels, ps_channels, out_channels, 4 * anchor_num)
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

        # self.transformer = Transformer(49*2, [128, 256, 512, 1024], 49*49)

    # def _transform_z(self, z_f, x_f, xf_crop=None, x_pos=None):
    #     if x_pos is not None:
    #         x_pos += 2.5
    #         x_pos = [s for s in x_pos]
    #         xf_crop = roi_align(x_f, x_pos, output_size=(7, 7))
    #     # template transform
    #     if xf_crop is not None:
    #         z_f_t = self.transformer(z_f, xf_crop)
    #         # z_f_t = 0.9 * z_f + 0.1 * z_f_t
    #         z_f_t = 0.5 * z_f + 0.5 * z_f_t
    #     else:
    #         z_f_t = z_f
    #     return z_f_t

    def forward(self, z_f, x_f, xf_crop=None, x_pos=None):
        # z_f_t = self._transform_z(z_f, x_f, xf_crop, x_pos)
        cls = self.cls(z_f, x_f, xf_crop, x_pos)
        loc = self.loc(z_f, x_f, xf_crop, x_pos)
        return cls, loc, x_f


class MultiRPN(RPN):
    def __init__(self, anchor_num, in_channels, out_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn'+str(i+2),
                    DepthwiseRPN(anchor_num, in_channels[i], out_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs, xf_crops, x_pos=None):
        cls = []
        loc = []
        # cls_head = []
        for idx, (z_f, x_f, xf_crop) in enumerate(zip(z_fs, x_fs, xf_crops), start=2):
            rpn = getattr(self, 'rpn'+str(idx))
            c, l, _ = rpn(z_f, x_f, xf_crop, x_pos)
            cls.append(c)
            loc.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight), x_fs
        else:
            return avg(cls), avg(loc), x_fs
