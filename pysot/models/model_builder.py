# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import RoIAlign, nms, roi_align

from pysot.config import cfg
from pysot.models.loss import weight_asp_loss, select_bce_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_rcnn_head
from pysot.models.neck import get_neck
from pysot.models.non_local import get_nonlocal
from pysot.models.backbone import mobilenet_v2_rcnn


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
        # self.roi_align_rcnn = RoIAlign((13, 13), 1.0 / cfg.ANCHOR.STRIDE, 1)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        rcnn_backbone = mobilenet_v2_rcnn(**cfg.BACKBONE.KWARGS)
        rcnn_pre_backbone = nn.Sequential( \
                nn.Conv2d(self.neck.out_channels, rcnn_backbone.in_channels, 1, bias=False), \
                nn.BatchNorm2d(rcnn_backbone.in_channels)
                )
        rcnn_post_backbone = nn.Sequential( \
                nn.Conv2d(320, cfg.RCNN.KWARGS['in_channels'], 1, bias=False), \
                nn.BatchNorm2d(cfg.RCNN.KWARGS['in_channels']), \
                nn.ReLU(inplace=True)
                )
        self.rcnn_backbone = nn.Sequential(rcnn_pre_backbone, rcnn_backbone, rcnn_post_backbone)

        # build rcnn head
        self.rcnn_head = get_rcnn_head(cfg.RCNN.TYPE,
                                       **cfg.RCNN.KWARGS)

    def template(self, z, roi_centered):
        zf = self.backbone(z)
        zf = self.neck(zf)
        # rcnn
        self.zf = self.crop_align_feature(zf, torch.Tensor(roi_centered).cuda())
        # self.zf = zf

    def track(self, x, anchor_cwh):
        '''
        anchors: [1, 4, 25, 25]
        '''
        xf = self.backbone(x)
        xf = self.neck(xf)

        # rpn
        # asp_rpn: [Nb, 1, 25, 25]
        ctr_rpn, asp_rpn = self.rpn_head(self.zf, xf)
        asp_rpn = torch.sqrt(torch.exp(asp_rpn))

        # each will be [1, 1, 25, 25]
        ax, ay, aw, ah = torch.split(anchor_cwh, 1, dim=1)

        aw = asp_rpn * aw.expand_as(asp_rpn)
        ah = asp_rpn * ah.expand_as(asp_rpn)

        # [Nb, 4, 25, 25]
        anchor = torch.cat([ \
            ax.expand_as(asp_rpn) - aw * 0.5, \
            ay.expand_as(asp_rpn) - ah * 0.5, \
            ax.expand_as(asp_rpn) + aw * 0.5, \
            ay.expand_as(asp_rpn) + ah * 0.5], dim=1)

        # assign target
        # for now we assume batch size of 1
        assert x.shape[0] == 1

        # nms
        anchor = torch.permute(torch.reshape(anchor, (4, -1)), (1 ,0))
        ctr_rpn = ctr_rpn.view(-1)
        keep = nms(anchor, ctr_rpn, 0.6667)[:cfg.RCNN.NUM_ROI]
        anchor = torch.index_select(anchor, 0, keep)

        # roi align
        # [num_roi, ch, 7, 7]
        xf_crop = self.crop_align_feature(xf, anchor.view(1, -1, 4))

        return {
                'ctr_rpn': ctr_rpn,
                'asp_rpn': asp_rpn
               }

    def crop_align_feature(self, feat, roi, out_shape):
        '''
        rois are assumed to be centered
        '''
        # rois are assumed to be centered, move it back.
        hh, ww = feat.shape[2:4]
        assert hh == ww
        offset = (hh / 2.0 - 0.5) * cfg.ANCHOR.STRIDE
        roi += offset

        scale = 1.0 / cfg.ANCHOR.STRIDE
        roi = tuple([r.view(-1, 4) for r in roi])
        if isinstance(feat, (list, tuple)):
            cropped = [roi_align(f, roi, out_shape, scale, 1) for f in feat]
        else:
            cropped = roi_align(feat, roi, out_shape, scale, 1)
        return cropped

    # def log_softmax(self, cls):
    #     b, a2, h, w = cls.size()
    #     if a2 == 2:
    #         cls = cls.permute(0, 2, 3, 1).contiguous()
    #         cls = F.log_softmax(cls, dim=3)
    #     else:
    #         cls = cls.view(b, a2//2, 2, h, w)
    #         cls = cls.permute(0, 1, 3, 4, 2).contiguous()
    #         cls = F.log_softmax(cls, dim=4)
    #     return cls

    # def log_sigmoid(self, ctr):
    #     '''
    #     stupid!
    #     '''
    #     ctr = ctr.permute(0, 2, 3, 1).contiguous()
    #     return ctr - torch.log1p(torch.exp(ctr))

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        template_box = data['template_box'].cuda()
        search_box = data['search_box'].cuda()

        label_ctr_rpn = data['ctr_rpn'].cuda()
        label_aspect_rpn = data['aspect_rpn'].cuda()
        label_aspect_w_rpn = torch.clamp(label_ctr_rpn, 0, 1)

        anchors_rcnn = data['anchors_rcnn'].cuda()
        anchors_cwh_rcnn = data['anchors_cwh_rcnn'].cuda()
        label_ctr_rcnn = data['ctr_rcnn'].cuda() # [Nb, num_roi]
        label_iou_rcnn = data['iou_rcnn'].cuda() # [Nb, num_roi]
        label_loc_rcnn = data['loc_rcnn'].cuda() # [Nb, 4, num_roi]

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        # neck with concat
        zf = self.neck(zf)
        xf = self.neck(xf)
        zf_crop = self.crop_align_feature(zf, template_box, (7, 7)) # (Nb, ch, 7, 7)

        # rpn head
        ctr_rpn, aspect_rpn = self.rpn_head(zf_crop, xf)

        # rcnn
        zf_crop_rcnn = self.crop_align_feature(zf, template_box, (15, 15))
        xf_crop_rcnn = self.crop_align_feature(xf, anchors_rcnn, (15, 15))

        zf_crop_rcnn = self.rcnn_backbone(zf_crop_rcnn)
        xf_crop_rcnn = self.rcnn_backbone(xf_crop_rcnn)

        # duplicate zf for cross correlation
        num_roi = xf_crop_rcnn.shape[0] // zf_crop_rcnn.shape[0]
        zf_crop_rcnn = torch.repeat_interleave(zf_crop_rcnn, num_roi, dim=0)

        # [Nb*num_roi, 1, 1, 1]
        # [Nb*num_roi, 1, 1, 1]
        # [Nb*num_roi, 4, 1, 1]
        ctr_rcnn, iou_rcnn, loc_rcnn = self.rcnn_head(zf_crop_rcnn, xf_crop_rcnn)

        # get loss
        ctr_rpn_loss = select_bce_loss(ctr_rpn, label_ctr_rpn)
        aspect_rpn_loss = (aspect_rpn - label_aspect_rpn.view(-1, 1, 1, 1)).abs()
        aspect_rpn_loss = torch.squeeze(aspect_rpn_loss) * label_aspect_w_rpn
        aspect_rpn_loss = aspect_rpn_loss.sum() / (label_aspect_w_rpn.sum() + 1e-08)

        ctr_rcnn_loss = F.binary_cross_entropy_with_logits(ctr_rcnn.view(-1), label_ctr_rcnn.view(-1))
        iou_rcnn_loss = F.binary_cross_entropy_with_logits(iou_rcnn.view(-1), label_iou_rcnn.view(-1))
        loc_rcnn_loss = (loc_rcnn.view(-1, 4) - label_loc_rcnn.view(-1, 4)).abs().sum(dim=1)
        loc_rcnn_loss *= label_ctr_rcnn.view(-1)
        loc_rcnn_loss = loc_rcnn_loss.sum() / (label_ctr_rcnn.sum() + 1e-08)

        outputs = {}
        outputs['total_loss'] = \
                cfg.TRAIN.CTR_WEIGHT * ctr_rpn_loss + \
                cfg.TRAIN.CTR_WEIGHT * aspect_rpn_loss + \
                cfg.TRAIN.CTR_WEIGHT * ctr_rcnn_loss + \
                cfg.TRAIN.CTR_WEIGHT * iou_rcnn_loss + \
                cfg.TRAIN.LOC_WEIGHT * loc_rcnn_loss
        outputs['ctr_rpn_loss'] = ctr_rpn_loss
        outputs['asp_rpn_loss'] = aspect_rpn_loss
        outputs['ctr_rcnn_loss'] = ctr_rcnn_loss
        outputs['iou_rcnn_loss'] = iou_rcnn_loss
        outputs['loc_rcnn_loss'] = loc_rcnn_loss
        
        # done
        return outputs
