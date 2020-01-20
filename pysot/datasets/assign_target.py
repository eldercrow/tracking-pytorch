# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import nms


class AssignTarget(object):
    '''
    '''
    def __init__(self, num_roi, scale, aspect):
        '''
        '''
        self.num_roi = num_roi
        self.scale = scale
        self.aspect = aspect

    def __call__(self, targets, anchors_cwh, ctr_rpn, asp_rpn, loc_rpn, gt_ctr_rpn):
        '''
        targets: roi, (Nb, 4)
        '''
        with torch.no_grad():
            # each will be [Nb, 1, 25, 25]
            ax, ay, aw, ah = torch.split(anchors_cwh, 1, dim=1)

            ax += (loc_rpn[:, 0:1, :, :] * aw)
            ay += (loc_rpn[:, 1:2, :, :] * ah)

            rand_asp_exp = torch.empty_like(asp_rpn)
            rand_asp_exp.normal_(mean=0, std=0.333)
            rand_scale_exp = torch.empty_like(asp_rpn)
            rand_scale_exp.normal_(mean=0, std=0.333)

            rand_asp = torch.ones_like(asp_rpn) * self.aspect
            rand_scale = torch.ones_like(asp_rpn) * self.scale

            rand_asp.pow_(rand_asp_exp)
            rand_scale.pow_(rand_scale_exp)

            asp = torch.sqrt(torch.exp(asp_rpn) * rand_asp)
            aw = aw.expand_as(asp_rpn) * asp * rand_scale
            ah = ah.expand_as(asp_rpn) / asp * rand_scale

            # [Nb, 4, 25, 25]
            anchor_all = torch.cat([ \
                ax.expand_as(asp) - aw * 0.5, \
                ay.expand_as(asp) - ah * 0.5, \
                ax.expand_as(asp) + aw * 0.5, \
                ay.expand_as(asp) + ah * 0.5], dim=1)

            # per-batch nms
            Nb = anchor_all.shape[0]

            # ctrs = ctr_rpn.view(Nb, -1)
            ctrs = gt_ctr_rpn.view(Nb, -1) + torch.sigmoid(ctr_rpn.view(Nb, -1))
            anchor_all = anchor_all.view(Nb, 4, -1).permute(0, 2, 1)

            ax = torch.reshape(ax, (Nb, -1,))
            ay = torch.reshape(ay, (Nb, -1,))
            aw = torch.reshape(aw, (Nb, -1,))
            ah = torch.reshape(ah, (Nb, -1,))

            selected_anchor = []
            selected_ctr = []
            selected_cls = []
            selected_loc = []

            for ii, (target, anchor, ctr) in enumerate(zip(targets, anchor_all, ctrs)):
                keep = nms(anchor, ctr, 0.7)[:(self.num_roi)]
                nms_anchor = torch.index_select(anchor, 0, keep)
                selected_anchor.append(nms_anchor)

                # for debug
                # ix = torch.min(target[2], nms_anchor[:, 2]) - torch.max(target[0], nms_anchor[:, 0])
                # iy = torch.min(target[3], nms_anchor[:, 3]) - torch.max(target[1], nms_anchor[:, 1])
                # inter = F.relu(ix) * F.relu(iy)
                # union = (target[2] - target[0]) * (target[3] - target[1]) + \
                #         (nms_anchor[:, 2] - nms_anchor[:, 0]) * (nms_anchor[:, 3] - nms_anchor[:, 1])
                # iou = inter / (union - inter)

                # (Nr, 4)
                loc = torch.stack([(ax[ii][keep] - target[0]) / aw[ii][keep], \
                                   (ay[ii][keep] - target[1]) / ah[ii][keep], \
                                   (target[2] - ax[ii][keep]) / aw[ii][keep], \
                                   (target[3] - ay[ii][keep]) / ah[ii][keep]], dim=1)
                selected_loc.append(loc)

            # stack anchor for roi_align
            selected_anchor = torch.stack(selected_anchor, dim=0) # [Nb, Nr, 4]

            # concat everything else
            # selected_ctr = torch.cat(selected_ctr, dim=0)
            selected_loc = F.relu(torch.cat(selected_loc, dim=0))

            # centerness
            dx0, dy0, dx1, dy1 = torch.split(selected_loc, 1, dim=1)
            selected_ctr = torch.sqrt((torch.min(dx0, dx1) / (torch.max(dx0, dx1) + 1e-08)) * \
                                      (torch.min(dy0, dy1) / (torch.max(dy0, dy1) + 1e-08)))
            selected_ctr = torch.reshape(selected_ctr, (-1,))

            selected_cls = -1 * torch.ones_like(selected_ctr, dtype=torch.int64)
            selected_cls[selected_ctr > 0.3] = 1
            selected_cls[selected_ctr <= 0] = 0

            return selected_ctr.detach(), \
                selected_cls.detach(), \
                selected_loc.detach(), \
                selected_anchor.detach()