# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F


def get_cls_loss(pred, label, select, weight=None):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return torch.Tensor((0.0,)).cuda()
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    if weight is None:
        return F.nll_loss(pred, label)
    else:
        w = torch.index_select(weight, 0, select)
        loss = F.nll_loss(pred, label, reduction='none')
        loss *= w
        return torch.sum(loss) / (torch.sum(w) + 1e-08)


def get_bce_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return torch.Tensor((0.0,)).cuda()
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.binary_cross_entropy_with_logits(pred, label)


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    nch = pred_loc.shape[1]
    # b = pred_loc.shape[0]
    pred_loc = pred_loc.view(-1, nch)
    label_loc = label_loc.view(-1, nch)
    loss_weight = loss_weight.view(-1)
    # b, _, sh, sw = pred_loc.size()
    # # pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1)
    loss = diff * loss_weight
    return loss.sum().div(loss_weight.sum() + 1e-08)


def weight_asp_loss(pred_asp, label_asp, loss_weight):
    b = pred_asp.shape[0]
    label_asp = label_asp.view(-1, 1, 1, 1)
    diff = (pred_asp - label_asp).abs()
    loss = diff.view(-1) * loss_weight.view(-1)
    return loss.sum().div(b)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def select_bce_loss(pred, label):
    pred = pred.view(-1)
    label = label.view(-1)
    pos = torch.logical_not(label.data.eq(-1)).nonzero().squeeze().cuda()
    loss_pos = get_bce_loss(pred, label, pos)
    return loss_pos
