# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.models.init_weight import init_weights

class Transformer(nn.Module):
    '''
    Pointnet-like feature transform sub-network.
    '''
    def __init__(self, ich, hiddens, och):
        '''
        '''
        super(Transformer, self).__init__()
        
        self.conv_in = self._conv_bn_relu(ich, ich, kernel_size=3)
        self.hiddens = nn.ModuleList(
            [self._conv_bn_relu(h0, h1, 1) for (h0, h1) in zip([ich] + hiddens[:-1], hiddens)]
        )
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.conv_out = nn.Conv2d(hiddens[-1], och, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(hiddens[-1], 2, kernel_size=1, bias=True)

    def _conv_bn_relu(self, ich, och, kernel_size, inplace=True):
        return nn.Sequential(
            nn.Conv2d(ich, och, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(och),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x, y):
        '''
        '''
        Nb, ch, hh, ww = x.shape
        och = hh*ww
        x = self.conv_in(x)
        y = self.conv_in(y)
        # swap channel and spatial
        x = x.permute(0, 2, 3, 1).reshape(Nb, hh*ww, 1, ch)
        y = y.permute(0, 2, 3, 1).reshape(Nb, hh*ww, 1, ch)
        # pointnet transformer
        p = self.conv_in(torch.cat([x, y], dim=1))
        for hidden in self.hiddens:
            p = hidden(p)
        p = self.pool(p)

        w = self.conv_w(p)
        w = F.softmax(w, dim=1)
        p = self.conv_out(p) # [Nb, (hh*ww)^2, 1, 1]
        
        # convert to groupwise conv kernel
        p = p.view(Nb, och, och, 1) # [N, hh*ww, hh*ww, 1]
        k = F.softmax(p, dim=2).view(Nb*och, och, 1, 1) # convex transform kernel
        # apply transform
        out = F.conv2d(x.reshape(1, Nb*hh*ww, 1, ch), k, groups=Nb)
        out = out.view(Nb, hh*ww, 1, ch).permute(0, 3, 1, 2).view(Nb, ch, hh, ww)
        return out, w