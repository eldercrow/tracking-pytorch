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
from pysot.models.non_local import get_nonlocal


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
        return {
                'cls': cls,
                'loc': loc,
                'ctr': ctr,
               }

    def log_softmax(self, cls):
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
        # template_box = data['template_box'].cuda()
        # search_box = data['search_box'].cuda()
        # 12: from template to search
        label_cls12 = data['label_cls12'].cuda()
        label_loc12 = data['label_loc12'].cuda()
        label_loc_weight12 = data['label_loc_weight12'].cuda()
        label_centerness12 = data['label_centerness12'].cuda()
        # 21: from search to template
        # label_cls21 = data['label_cls21'].cuda()
        # label_loc21 = data['label_loc21'].cuda()
        # label_loc_weight21 = data['label_loc_weight21'].cuda()
        # label_centerness21 = data['label_centerness21'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        # neck
        zf = self.neck(zf)
        xf = self.neck(xf)
        # non-local
        # zf = self.non_local(zf)
        # xf = self.non_local(xf)

        # crop
        # first adjust coordinate
        # hh, ww = zf[0].shape[2:4] if isinstance(zf, (list, tuple)) else zf.shape[2:4]
        # assert hh == ww
        # offset = (hh / 2.0 - 0.5) * cfg.ANCHORLESS.STRIDE
        # template_box += offset
        # search_box += offset
        #
        # template_box = torch.split(template_box, 1, dim=0)
        # search_box = torch.split(search_box, 1, dim=0)

        # head
        cls12, loc12, ctr12 = self.rpn_head(zf, xf)
        # cls21, loc21, ctr21 = self.rpn_head(xf_crop, zf)

        # get loss
        cls12 = self.log_softmax(cls12)
        cls_loss = select_cross_entropy_loss(cls12, label_cls12, label_centerness12)
        loc_loss = weight_l1_loss(loc12, label_loc12, label_loc_weight12)
        # ctr12 = torch.sigmoid(ctr12)
        ctr_loss = select_bce_loss(ctr12, label_centerness12)

        # cls21 = self.log_softmax(cls21)
        # cls_loss21 = select_cross_entropy_loss(cls21, label_cls21, label_centerness21)
        # loc_loss21 = weight_l1_loss(loc21, label_loc21, label_loc_weight21)
        # ctr21 = torch.sigmoid(ctr21)
        # ctr_loss21 = select_bce_loss(ctr21, label_centerness12)

        # cls_loss = 0.5 * (cls_loss12 + cls_loss21)
        # loc_loss = 0.5 * (loc_loss12 + loc_loss21)
        # ctr_loss = 0.5 * (ctr_loss12 + ctr_loss21)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + \
            cfg.TRAIN.CTR_WEIGHT * ctr_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['ctr_loss'] = ctr_loss
        # done
        return outputs
