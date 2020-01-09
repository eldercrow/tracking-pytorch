# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import RoIAlign, nms

from pysot.config import cfg
from pysot.models.loss import weight_asp_loss, select_bce_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_rcnn_head
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
        self.roi_align = RoIAlign((7, 7), 1.0 / cfg.ANCHOR.STRIDE, 1)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build rcnn head
        # self.rcnn_head = get_rcnn_head(cfg.RCNN.TYPE,
        #                                **cfg.RCNN.KWARGS)

    def template(self, z, roi_centered):
        zf = self.backbone(z)
        zf = self.neck(zf)
        # rcnn
        self.zf_crop = self.crop_align_feature(zf, torch.Tensor(roi_centered).cuda())
        # rpn
        l = (zf.shape[-1] - cfg.ADJUST.CROP_SIZE) // 2
        r = l + cfg.ADJUST.CROP_SIZE
        zf = zf[:, :, l:r, l:r]
        self.zf = zf

    def track(self, x, anchors):
        '''
        anchors: [num_anchor, 4]
        '''
        xf = self.backbone(x)
        xf = self.neck(xf)

        # rpn
        cls_rpn, ctr_rpn = self.rpn_head(self.zf, xf)
        # get max scores
        cls_rpn = self.log_softmax(cls_rpn)
        ctr_rpn = self.log_sigmoid(ctr_rpn)
        shape_ctr = ctr_rpn.shape
        ctr_rpn = ctr_rpn.view(*([shape_ctr[0], 1] + [s for s in shape_ctr[1:]]))
        score_rpn = cls_rpn[:, :] + ctr_rpn.expand_as(cls_rpn)
        score_rpn = score_rpn[:, :, :, :, 1].contiguous()

        Nb = xf.shape[0]
        score_rpn = score_rpn.view(Nb, -1)

        # nms
        anchors = torch.Tensor(anchors).cuda()
        rois = []
        for score in score_rpn:
            keep = nms(anchors, score, 0.6667)[:cfg.RCNN.NUM_ROI]
            rois.append(torch.index_select(anchors, 0, keep))
        rois = torch.stack(rois, 0)

        # rcnn
        zf_crop = torch.repeat_interleave(self.zf_crop, cfg.RCNN.NUM_ROI, dim=0)
        xf_crop = self.crop_align_feature(xf, rois)
        cls_rcnn, loc_rcnn, ctr_rcnn = self.rcnn_head(zf_crop, xf_crop)

        return {
                'cls': cls_rcnn,
                'loc': loc_rcnn,
                'ctr': ctr_rcnn,
                'roi': rois,
                'cls_rpn': cls_rpn,
                'ctr_rpn': ctr_rpn
               }

    def crop_align_feature(self, feat, roi):
        '''
        rois are assumed to be centered
        '''
        # rois are assumed to be centered, move it back.
        hh, ww = feat.shape[2:4]
        assert hh == ww
        offset = (hh / 2.0 - 0.5) * cfg.ANCHOR.STRIDE
        roi += offset

        roi = tuple([r.view(-1, 4) for r in roi])
        if isinstance(feat, (list, tuple)):
            cropped = [self.roi_align(f, roi) for f in feat]
        else:
            cropped = self.roi_align(feat, roi)
        return cropped

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        if a2 == 2:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        else:
            cls = cls.view(b, a2//2, 2, h, w)
            cls = cls.permute(0, 1, 3, 4, 2).contiguous()
            cls = F.log_softmax(cls, dim=4)
        return cls

    def log_sigmoid(self, ctr):
        '''
        stupid!
        '''
        ctr = ctr.permute(0, 2, 3, 1).contiguous()
        return ctr - torch.log1p(torch.exp(ctr))

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        template_box = data['template_box'].cuda()
        search_box = data['search_box'].cuda()

        # 12: from template to search
        label_ctr_rpn = data['label_ctr_rpn'].cuda()
        label_aspect_rpn = data['label_aspect_rpn'].cuda()
        label_aspect_w_rpn = data['label_aspect_w_rpn'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        # neck with concat
        zf = self.neck(zf)
        xf = self.neck(xf)

        # for rcnn
        zf_crop = self.crop_align_feature(zf, template_box) # (Nb, ch, 7, 7)

        # rpn head
        ctr_rpn, aspect_rpn = self.rpn_head(zf_crop, xf)

        # get loss
        ctr_rpn_loss = select_bce_loss(ctr_rpn, label_ctr_rpn)
        aspect_rpn_loss = weight_asp_loss(aspect_rpn, label_aspect_rpn, label_aspect_w_rpn)

        outputs = {}
        outputs['total_loss'] = \
            cfg.TRAIN.CTR_WEIGHT * ctr_rpn_loss + \
            cfg.TRAIN.LOC_WEIGHT * aspect_rpn_loss
        outputs['ctr_rpn_loss'] = ctr_rpn_loss
        outputs['aspect_rpn_loss'] = aspect_rpn_loss
        
        # done
        return outputs
