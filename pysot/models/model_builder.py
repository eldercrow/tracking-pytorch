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
from pysot.models.loss import weight_asp_loss, select_bce_loss, select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_rcnn_head
from pysot.models.neck import get_neck
from pysot.models.non_local import get_nonlocal
from pysot.models.backbone import mobilenet_v2_rcnn
from pysot.datasets.assign_target import AssignTarget


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

        rcnn_backbone = mobilenet_v2_rcnn(**cfg.BACKBONE.KWARGS)
        rcnn_pre_backbone = nn.Sequential( \
                nn.Conv2d(self.neck.out_channels, rcnn_backbone.in_channels, 1, bias=False), \
                nn.BatchNorm2d(rcnn_backbone.in_channels)
                )
        # rcnn_post_backbone = nn.Sequential( \
        #         nn.Conv2d(320, cfg.RCNN.KWARGS['in_channels'], 1, bias=False), \
        #         nn.BatchNorm2d(cfg.RCNN.KWARGS['in_channels']), \
        #         nn.ReLU(inplace=True)
        #         )
        self.rcnn_backbone = nn.Sequential(rcnn_pre_backbone, rcnn_backbone)

        # build rcnn head
        self.rcnn_head = get_rcnn_head(cfg.RCNN.TYPE,
                                       **cfg.RCNN.KWARGS)

        self.assign_target = AssignTarget(cfg.TRAIN.NUM_ROI)

    def template(self, z, roi_centered):
        zf = self.backbone(z)
        zf = self.neck(zf)

        # rcnn
        roi = torch.Tensor(roi_centered).cuda()
        self.zf_rpn = self.crop_align_feature(zf, roi, (7, 7))

        # zf_rcnn = self.crop_align_feature(zf, roi, (11, 11))
        # zf_rcnn = self.rcnn_backbone(zf_rcnn)
        zf_rcnn = zf[:, :, 4:11, 4:11].contiguous()
        self.zf_rcnn = torch.repeat_interleave(zf_rcnn, cfg.RCNN.NUM_ROI, dim=0)

        # self.zf = zf

    def track(self, x, anchor_cwh):
        '''
        anchors: [1, 4, 25, 25]
        '''
        xf = self.backbone(x)
        xf = self.neck(xf)

        # rpn
        # asp_rpn: [Nb, 1, 25, 25]
        ctr_rpn, asp_rpn, loc_rpn = self.rpn_head(self.zf_rpn, xf)

        # each will be [1, 1, 25, 25]
        ax, ay, aw, ah = torch.split(anchor_cwh, 1, dim=1)

        ax += (loc_rpn[:, :1, :, :] * aw)
        ay += (loc_rpn[:, 1:, :, :] * ah)

        asp_rpn = torch.sqrt(torch.exp(asp_rpn))
        aw = aw.expand_as(asp_rpn) * asp_rpn
        ah = ah.expand_as(asp_rpn) / asp_rpn

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
        anchor = anchor.view(4, -1).permute(1, 0).contiguous()
        ctr_rpn = ctr_rpn.view(-1)
        keep = nms(anchor, torch.sigmoid(ctr_rpn), 0.7)[:cfg.RCNN.NUM_ROI]
        anchor = torch.index_select(anchor, 0, keep)

        # roi align
        # [num_roi, ch, 7, 7]
        xf_rcnn = self.crop_align_feature(xf, anchor.view(1, -1, 4), (11, 11))
        xf_rcnn = self.rcnn_backbone(xf_rcnn)

        ctr_rcnn, cls_rcnn, loc_rcnn = self.rcnn_head(self.zf_rcnn, xf_rcnn)
        ctr_rcnn = 0.5 * (ctr_rcnn.view(-1) + ctr_rpn[keep])

        return {
                'ctr_rpn': ctr_rpn,
                'asp_rpn': asp_rpn,
                'ctr_rcnn': ctr_rcnn,
                'cls_rcnn': cls_rcnn,
                'loc_rcnn': loc_rcnn,
                'roi_rcnn': anchor
               }

    def crop_align_feature(self, feat, roi_centered, out_shape):
        '''
        rois are assumed to be centered
        '''
        # rois are assumed to be centered, move it back.
        hh, ww = feat.shape[2:4]
        assert hh == ww
        offset = (hh / 2.0 - 0.5) * cfg.ANCHOR.STRIDE
        roi = roi_centered + offset

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
        gt_ctr_rpn = data['ctr_rpn_all'].cuda()
        label_aspect_rpn = data['aspect_rpn'].cuda()
        label_loc_rpn = data['loc_rpn'].cuda()
        anchors_cwh = data['anchors_cwh'].cuda()
        label_aspect_w_rpn = torch.clamp(label_ctr_rpn, 0, 1)

        # anchors_rcnn = data['anchors_rcnn'].cuda()
        # anchors_cwh_rcnn = data['anchors_cwh_rcnn'].cuda()
        # label_ctr_rcnn = data['ctr_rcnn'].cuda() # [Nb, num_roi]
        # label_iou_rcnn = data['iou_rcnn'].cuda() # [Nb, num_roi]
        # label_loc_rcnn = data['loc_rcnn'].cuda() # [Nb, 4, num_roi]

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        # neck with concat
        zf = self.neck(zf)
        xf = self.neck(xf)
        zf_rpn = self.crop_align_feature(zf, template_box, (7, 7)) # (Nb, ch, 7, 7)

        # rpn head
        ctr_rpn, aspect_rpn, loc_rpn = self.rpn_head(zf_rpn, xf)

        label_ctr_rcnn, label_cls_rcnn, label_loc_rcnn, anchors_rcnn = \
                self.assign_target(search_box, anchors_cwh, ctr_rpn, aspect_rpn, loc_rpn, gt_ctr_rpn)

        # rcnn
        zf_rcnn = zf[:, :, 4:11, 4:11] #self.crop_align_feature(zf, template_box, (11, 11))
        xf_rcnn = self.crop_align_feature(xf, anchors_rcnn, (11, 11))

        # zf_rcnn = self.rcnn_backbone(zf_rcnn)
        xf_rcnn = self.rcnn_backbone(xf_rcnn)

        # duplicate zf for cross correlation
        num_roi = xf_rcnn.shape[0] // zf_rcnn.shape[0]
        assert num_roi == cfg.TRAIN.NUM_ROI
        zf_rcnn = torch.repeat_interleave(zf_rcnn, num_roi, dim=0)

        # [Nb*num_roi, 1, 1, 1]
        # [Nb*num_roi, 1, 1, 1]
        # [Nb*num_roi, 4, 1, 1]
        ctr_rcnn, cls_rcnn, loc_rcnn = self.rcnn_head(zf_rcnn, xf_rcnn)
        cls_rcnn = F.log_softmax(cls_rcnn.view(-1, 2), dim=1)

        # get loss
        ctr_rpn_loss = select_bce_loss(ctr_rpn, label_ctr_rpn)
        aspect_rpn_loss = (aspect_rpn - label_aspect_rpn.view(-1, 1, 1, 1)).abs()
        aspect_rpn_loss = torch.squeeze(aspect_rpn_loss) * label_aspect_w_rpn
        aspect_rpn_loss = aspect_rpn_loss.sum() / (label_aspect_w_rpn.sum() + 1e-08)
        loc_rpn_loss = weight_l1_loss(loc_rpn, label_loc_rpn, label_aspect_w_rpn)

        ctr_rcnn_loss = F.binary_cross_entropy_with_logits(ctr_rcnn.view(-1), label_ctr_rcnn.view(-1))
        cls_rcnn_loss = select_cross_entropy_loss(cls_rcnn, label_cls_rcnn.view(-1))
        loc_rcnn_loss = (loc_rcnn.view(-1, 4) - label_loc_rcnn.view(-1, 4)).abs().sum(dim=1)
        loc_rcnn_loss *= label_ctr_rcnn.view(-1)
        loc_rcnn_loss = loc_rcnn_loss.sum() / (label_ctr_rcnn.sum() + 1e-08)

        outputs = {}
        outputs['total_loss'] = \
                cfg.TRAIN.CTR_WEIGHT * ctr_rpn_loss + \
                cfg.TRAIN.CTR_WEIGHT * aspect_rpn_loss + \
                cfg.TRAIN.LOC_WEIGHT * loc_rpn_loss + \
                cfg.TRAIN.CTR_WEIGHT * ctr_rcnn_loss + \
                cfg.TRAIN.CTR_WEIGHT * cls_rcnn_loss + \
                cfg.TRAIN.LOC_WEIGHT * loc_rcnn_loss
        outputs['ctr_rpn_loss'] = ctr_rpn_loss
        outputs['asp_rpn_loss'] = aspect_rpn_loss
        outputs['loc_rpn_loss'] = loc_rpn_loss
        outputs['ctr_rcnn_loss'] = ctr_rcnn_loss
        outputs['cls_rcnn_loss'] = cls_rcnn_loss
        outputs['loc_rcnn_loss'] = loc_rcnn_loss
        
        # done
        return outputs
