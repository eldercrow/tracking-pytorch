# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision.ops import RoIAlign

from pysot.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss, select_bce_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head
from pysot.models.neck import get_neck
# from pysot.models.non_local import get_nonlocal


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        self.neck = get_neck(cfg.ADJUST.TYPE,
                             **cfg.ADJUST.KWARGS)

        # roi align for cropping center
        # self.roi_align = RoIAlign((7, 7), 1.0 / cfg.ANCHOR.STRIDE, 1)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

    def template(self, z):
        zf = self.backbone(z)
        zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        xf = self.neck(xf)
        cls, loc, ctr = self.rpn_head(self.zf, xf)
        # cls = self.weighted_sum(cls, cls_weight)
        # loc = self.weighted_sum_loc(loc, loc_weight)
        # ctr = self.weighted_sum(ctr, ctr_weight)
        return {
                'cls': cls,
                'loc': loc,
                'ctr': ctr,
               }

    def log_softmax(self, cls):
        assert cls.size(1) == 2
        cls = cls.permute(0, 2, 3, 1).contiguous()
        cls = F.log_softmax(cls, dim=3)
        # b, a2, h, w = cls.size()
        # cls = cls.view(b, 2, a2//2, h, w)
        # cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        # cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        # 12: from template to search
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        label_ctr = data['label_ctr'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        # neck
        zf = self.neck(zf)
        xf = self.neck(xf)

        # head
        cls, loc, ctr = self.rpn_head(zf, xf)
        cls = self.log_softmax(cls)

        # get loss
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
        # ctr12 = torch.sigmoid(ctr12)
        ctr_loss = select_bce_loss(ctr, label_ctr)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + \
            cfg.TRAIN.CTR_WEIGHT * ctr_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['ctr_loss'] = ctr_loss
        # done
        return outputs
