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
        return torch.tensor(0)
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
        return torch.tensor(0)
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    # return F.binary_cross_entropy(pred, label)
    return F.binary_cross_entropy_with_logits(pred, label)


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    # pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1)
    loss = diff * loss_weight
    return loss.sum().div(b)


def weight_giou_loss(pred_loc, label_loc, loss_weight):
    Nb = pred_loc.shape[0]

    x0g, y0g, x1g, y1g = torch.split(label_loc, 1, dim=1)
    x0p, y0p, x1p, y1p = torch.split(pred_loc, 1, dim=1)

    x0g *= -1
    y0g *= -1
    x0p *= -1
    y0p *= -1

    gt_area = (x1g - x0g) * (y1g - y0g)
    pr_area = (x1p - x0p) * (y1p - y0p)

    # iou
    ix0 = torch.max(x0g, x0p)
    iy0 = torch.max(y0g, y0p)
    ix1 = torch.min(x1g, x1p)
    iy1 = torch.min(y1g, y1p)
    inter = torch.clamp(ix1 - ix0, min=0) * torch.clamp(iy1 - iy0, min=0)

    union = gt_area + pr_area - inter
    iou = inter / torch.clamp(union, min=1e-08)

    # enclosure
    ex0 = torch.min(x0g, x0p)
    ey0 = torch.min(y0g, y0p)
    ex1 = torch.max(x1g, x1p)
    ey1 = torch.max(y1g, y1p)
    enclosure = torch.clamp(ex1 - ex0, min=0) * torch.clamp(ey1 - ey0, min=0)

    giou = iou - (enclosure - union) / torch.clamp(enclosure, min=1e-08)
    loss = torch.squeeze(1. - giou) * loss_weight
    return loss.sum().div(Nb)


def select_cross_entropy_loss(pred, label, weight=None):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    # weight = weight.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    # loss_pos = get_bce_loss(pred, label, pos) #, weight)
    # loss_neg = get_bce_loss(pred, label, neg)
    loss_pos = get_cls_loss(pred, label, pos, weight)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def select_bce_loss(pred, label):
    pred = pred.view(-1)
    label = label.view(-1)
    pos = torch.logical_not(label.data.eq(-1)).nonzero().squeeze().cuda()
    loss_pos = get_bce_loss(pred, label, pos)
    return loss_pos
