import torch
import numpy as np
from torch import nn
import math

from pysot.models.head.xcorr import xcorr_fast, xcorr_depthwise #, L1DiffFunction


class CARHead(torch.nn.Module):
    def __init__(self, in_channels, hiddens, num_convs):
        """
        Arguments:
            in_channels (int or list of ints): number of channels of the input feature
            hiddens: number of channels used in the module
        """
        super(CARHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = 2 #cfg.TRAIN.NUM_CLASSES

        self.in_channels = in_channels

        if isinstance(in_channels, (list, tuple)):
            ich = np.sum(in_channels)
        else:
            ich = in_channels

        cls_tower = []
        bbox_tower = []
        for i in range(num_convs):
            cls_tower.append(
                nn.Conv2d(
                    ich if i == 0 else hiddens,
                    hiddens,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, hiddens))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    ich if i == 0 else hiddens,
                    hiddens,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, hiddens))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            hiddens, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            hiddens, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            hiddens, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        # prior_prob = cfg.TRAIN.PRIOR_PROB
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, xf, zf):
        #
        if isinstance(xf, (list, tuple)):
            assert isinstance(zf, (list, tuple))
            for ii in range(len(xf)):
                assert self.in_channels[ii] == xf[ii].size(1)
                assert self.in_channels[ii] == zf[ii].size(1)
            x = [xcorr_depthwise(zi, xi) for (zi, xi) in zip(zf, xf)]
            x = torch.cat(x, dim=1)
        else:
            assert self.in_channels == xf.size(1)
            x = xcorr_depthwise(zf, xf)

        cls_tower = self.cls_tower(x)
        logits = self.cls_logits(cls_tower)
        centerness = self.centerness(cls_tower)
        bbox_reg = torch.exp(self.bbox_pred(self.bbox_tower(x)))

        return logits, bbox_reg, centerness


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale