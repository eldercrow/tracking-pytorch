# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F


def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.reshape(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


def xcorr_circular(x, kernel):
    '''
    '''
    Nb, nch, hh, ww = x.shape
    assert kernel.size(2) == 1 and kernel.size(3) == 1
    kernel = kernel.expand_as(x)

    x = torch.reshape(x.permute(0, 2, 3, 1), (1, Nb*hh*ww, 1, nch))
    x = F.pad(x, (0, nch-1, 0, 0), mode='circular')
    kernel = torch.reshape(kernel.permute(0, 2, 3, 1), (Nb*hh*ww, 1, 1, nch))

    res = F.conv2d(x, kernel, groups=Nb*hh*ww)
    res = torch.reshape(res, (Nb, hh, ww, nch)).permute(0, 3, 1, 2)
    return res

# # Inherit from Function
# class L1DiffFunction(torch.autograd.Function):
#
#     # Note that both forward and backward are @staticmethods
#     @staticmethod
#     # bias is an optional argument
#     def forward(ctx, lhs, rhs):
#         diff = lhs - rhs
#         ctx.save_for_backward(lhs, rhs, diff)
#         return diff
#
#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         # This is a pattern that is very convenient - at the top of backward
#         # unpack saved_tensors and initialize all gradients w.r.t. inputs to
#         # None. Thanks to the fact that additional trailing Nones are
#         # ignored, the return statement is simple even when the function has
#         # optional inputs.
#         lhs, rhs, diff = ctx.saved_tensors
#         grad_lhs = grad_rhs = None
#
#         # These needs_input_grad checks are optional and there only to
#         # improve efficiency. If you want to make your code simpler, you can
#         # skip them. Returning gradients for inputs that don't require it is
#         # not an error.
#         if ctx.needs_input_grad[0]:
#             grad_lhs = grad_output * diff
#         if ctx.needs_input_grad[1]:
#             grad_rhs = -grad_output * diff
#
#         return grad_lhs, grad_rhs
