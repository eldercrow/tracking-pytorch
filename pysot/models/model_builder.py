# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from pysot.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head
from pysot.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        self.neck = get_neck(cfg.ADJUST.TYPE,
                             **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x, xf_crops=None):
        xf = self.backbone(x)
        xf = self.neck(xf)
        cls, loc, _ = self.rpn_head(self.zf, xf, xf_crops=xf_crops)
        return {
                'cls': cls,
                'loc': loc,
                'xf': xf
               }

    # def track_ls(self, x, ft1=None):
    #     assert not cfg.MASK.MASK
    #     xf = self.backbone(x)
    #     if cfg.ADJUST.ADJUST:
    #         xf = self.neck(xf)

    #     if ft1 is not None:
    #         self.zf[-1] = ft1

    #     cls, loc = self.rpn_head(self.zf, xf)
    #     return {
    #             'cls': cls,
    #             'loc': loc,
    #             'ft': xf[-1]
    #            }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        x_pos = data['x_pos'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        # neck
        zf = self.neck(zf)
        xf = self.neck(xf)
        x_crops = [None for _ in xf] if isinstance(xf, (list, tuple)) else None
        # head
        cls, loc, _ = self.rpn_head(zf, xf, x_crops, x_pos)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        # done
        return outputs
