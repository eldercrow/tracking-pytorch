import torch
import numpy as np
from torch import nn
import math

from pysot.models.head.xcorr import xcorr_circular #, L1DiffFunction
from pysot.models.init_weight import init_weights


class CorrHead(torch.nn.Module):
    def __init__(self, in_channels, hiddens):
        """
        Arguments:
            in_channels (int or list of ints): number of channels of the input feature
            hiddens: number of channels used in the module
        """
        super(CorrHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = 2 #cfg.TRAIN.NUM_CLASSES

        self.in_channels = in_channels

        def _conv3_gn_relu(ich, och, kernel_size, **kwargs):
            return nn.Sequential(
                nn.Conv2d(ich, och, kernel_size, **kwargs),
                nn.GroupNorm(32, och),
                nn.ReLU()
            )

        if isinstance(in_channels, (list, tuple)):
            # search
            for ii, ich in enumerate(in_channels):
                layers = []
                for _ in range(3):
                    layers.append(nn.Conv2d(ich, ich, 3, padding=0, bias=False))
                    layers.append(nn.GroupNorm(32, ich))
                    layers.append(nn.ReLU())
                self.add_module('preproc_x{}'.format(ii+2), nn.Sequential(*layers))
            # target
            for ii, ich in enumerate(in_channels):
                layers = []
                for _ in range(3):
                    layers.append(nn.Conv2d(ich, ich, 3, padding=0, bias=False))
                    layers.append(nn.GroupNorm(32, ich))
                    layers.append(nn.ReLU())
                self.add_module('preproc_z{}'.format(ii+2), nn.Sequential(*layers))
            ich_all = np.sum(in_channels)
        else:
            assert False, 'not implemented'

        self.cls_tower = nn.Sequential(
            nn.Conv2d(ich_all, hiddens, 3, padding=1, bias=False),
            nn.GroupNorm(32, hiddens),
            nn.ReLU()
        )
        self.bbox_tower = nn.Sequential(
            nn.Conv2d(ich_all, hiddens, 3, padding=1, bias=False),
            nn.GroupNorm(32, hiddens),
            nn.ReLU()
        )

        self.cls_logits = nn.Sequential(
            nn.Conv2d(hiddens, hiddens, 1, bias=False),
            nn.GroupNorm(32, hiddens),
            nn.ReLU(),
            nn.Conv2d(hiddens, num_classes, 3, padding=1)
        )
        self.bbox_pred = nn.Sequential(
            nn.Conv2d(hiddens, hiddens, 1, bias=False),
            nn.GroupNorm(32, hiddens),
            nn.ReLU(),
            nn.Conv2d(hiddens, 4, 3, padding=1)
        )
        self.centerness = nn.Sequential(
            nn.Conv2d(hiddens, hiddens, 1, bias=False),
            nn.GroupNorm(32, hiddens),
            nn.ReLU(),
            nn.Conv2d(hiddens, 1, 3, padding=1)
        )

        init_weights(self)
        # # initialization
        # for modules in [self.cls_tower, self.bbox_tower,
        #                 self.cls_logits, self.bbox_pred,
        #                 self.centerness]:
        #     for l in modules.modules():
        #         if isinstance(l, nn.Conv2d):
        #             torch.nn.init.normal_(l.weight, std=0.01)
        #             torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        # prior_prob = cfg.TRAIN.PRIOR_PROB
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, zf, xf):
        #
        if isinstance(xf, (list, tuple)):
            assert isinstance(zf, (list, tuple))
            x = []
            for ii, (xi, zi) in enumerate(zip(xf, zf)):
                xi = getattr(self, 'preproc_x{}'.format(ii+2))(xi)
                zi = getattr(self, 'preproc_z{}'.format(ii+2))(zi)
                x.append(xcorr_circular(xi, zi))
            x = torch.cat(x, dim=1)
        else:
            assert False, 'not implemented'

        cls_tower = self.cls_tower(x)
        logits = self.cls_logits(cls_tower)
        centerness = self.centerness(cls_tower)
        bbox_reg = self.bbox_pred(self.bbox_tower(x))
        # bbox_reg = torch.exp(bbox_reg)

        return logits, bbox_reg, centerness


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
