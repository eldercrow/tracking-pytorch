'''
Modified from https://github.com/PkuRainBow/OCNet.pytorch/blob/master/oc_module/base_oc_block.py
'''

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


# class PSPModule(nn.Module):
#     # (1, 2, 3, 6)
#     def __init__(self, sizes=(1,3,7), dimension=2):
#         super(PSPModule, self).__init__()
#         self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

#     def _make_stage(self, size, dimension=2):
#         if dimension == 1:
#             prior = nn.AdaptiveAvgPool1d(output_size=size)
#         elif dimension == 2:
#             prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
#         elif dimension == 3:
#             prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
#         return prior

#     def forward(self, feats):
#         n, c, _, _ = feats.size()
#         priors = [stage(feats).view(n, c, -1) for stage in self.stages]
#         center = torch.cat(priors, -1)
#         return center


class PSPModule(nn.Module):
    def __init__(self, in_hw, pool_iter=1, sigma=0.2):
        super(PSPModule, self).__init__()
        self.pool = nn.MaxPool2d(3, 2, padding=0)
        self.pool_iter = pool_iter

        # weight map
        hh, ww = in_hw
        X0, Y0 = self._create_grid(hh, ww)

        Xp = []
        Yp = []
        for _ in range(pool_iter):
            hh //= 2
            ww //= 2
            X, Y = self._create_grid(hh, ww)
            Xp.append(X)
            Yp.append(Y)
        Xp = np.hstack(Xp)
        Yp = np.hstack(Yp)

        dx = np.reshape(X0, (-1, 1)) - np.reshape(Xp, (1, -1))
        dy = np.reshape(Y0, (-1, 1)) - np.reshape(Yp, (1, -1))
        d2 = np.expand_dims(dx*dx + dy*dy, 0)

        self.P0 = torch.tensor(np.expand_dims(np.stack([X0, Y0], axis=1), 0))
        self.Pp = torch.tensor(np.expand_dims(np.stack([Xp, Yp], axis=1), 0))
        self.logW = torch.tensor(-0.5 * d2 / sigma / sigma)

    def _create_grid(self, hh, ww):
        X, Y = np.meshgrid(np.arange(ww), np.arange(hh))
        X = X.astype(np.float32) / ww
        Y = Y.astype(np.float32) / hh
        return X.ravel(), Y.ravel()

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = []
        prior = feats
        for _ in range(self.pool_iter):
            prior = self.pool(prior)
            priors.append(prior)
        priors = [p.view(n, c, -1) for p in priors]
        center = torch.cat(priors, -1)
        return center


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels, in_hw, pool_iter=1):
        super(_SelfAttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        # if out_channels == None:
        #     self.out_channels = in_channels
        # if pool_iter > 0:
        #     self.pool = nn.MaxPool2d(3, 2, padding=0)
        # self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

        self.psp = PSPModule(in_hw, pool_iter)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        value = self.psp(self.f_value(x))

        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x)
        # value=self.psp(value)#.view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        key = self.psp(key)  # .view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map + self.psp.logW, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)

        dp = self.psp.P0 - torch.matmul(sim_map, self.psp.Pp)
        dp = dp.view(batch_size, 2, *x.size()[2:])
        return torch.cat([context, dp], dim=1)


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels, in_hw, pool_iter=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   in_hw,
                                                   pool_iter)


class APNB(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        pool_iter: max pooling iteration, kernel_size=3, stride=2, padding=0
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, in_hw, pool_iter=2, dropout=0.05):
        super(APNB, self).__init__()
        self.stage = self._make_stage(in_channels, out_channels, key_channels, value_channels, in_hw, pool_iter)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels + 2, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, in_hw, pool_iter):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    in_hw,
                                    pool_iter)

    def forward(self, feats):
        context = self.stage(feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output
