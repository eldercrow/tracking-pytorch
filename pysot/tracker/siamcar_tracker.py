# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch

from pysot.config import cfg
from pysot.utils.anchor import Anchors, generate_anchor
from pysot.utils.bbox import IoU, center2corner
from pysot.tracker.base_tracker import SiameseTracker


class SiamCARTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamCARTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        anchors, anchors_cwh = generate_anchor(self.score_size, \
                                               cfg.ANCHOR.SCALES, \
                                               cfg.ANCHOR.RATIOS, \
                                               cfg.ANCHOR.STRIDE)

        self.anchors_cwh = torch.Tensor(np.expand_dims(anchors_cwh, 0)).cuda()
        self.model = model
        self.model.eval()

    def _convert_bbox(self, delta, roi):
        '''
        delta: [Nb*num_roi, 4, 1, 1]
        both delta and anchor should be [N, 4]
        '''
        roi = roi.view(-1, 4).data.cpu().numpy()
        ax = (roi[:, 0] + roi[:, 2]) * 0.5
        ay = (roi[:, 1] + roi[:, 3]) * 0.5
        aw = roi[:, 2] - roi[:, 0]
        ah = roi[:, 3] - roi[:, 1]

        delta = delta.view(-1, 4).data.cpu().numpy()
        delta[:, 0] = ax - delta[:, 0] * aw
        delta[:, 1] = ay - delta[:, 1] * ah
        delta[:, 2] = ax + delta[:, 2] * aw
        delta[:, 3] = ay + delta[:, 3] * ah
        return delta

    # def _convert_score(self, score):
    #     '''
    #     score: [Nb*num_roi, 2, 1, 1]
    #     '''
    #     score = score.view(-1, 2)
    #     score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
    #     return score

    def _convert_centerness(self, ctr):
        ctr = torch.sigmoid(ctr.view(-1)).data.cpu().numpy()
        return ctr

    def _convert_aspect(self, aspect):
        asp = aspect.data.cpu().numpy()
        return asp

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.bbox0 = [b for b in bbox]
        self.center_pos = [bbox[0] + bbox[2]*0.5, bbox[1] + bbox[3]*0.5]
        self.center_pos = np.array(self.center_pos)
        # self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
        #                             bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.array(cfg.PREPROC.PIXEL_MEAN[::-1])
        # self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        roi = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        roi_centered = np.array(roi) - np.tile(self.center_pos, (2,))
        roi_centered = np.reshape(roi_centered, (1, 1, 4))

        self.model.template(z_crop, roi_centered)
        self.roi_centered = roi_centered

        # zf_rpn = self.model.zf_rpn
        # zf_rcnn = self.model.zf_rcnn
        # zf_crop = self.model.zf_crop

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        # 'cls': cls,
        # 'loc': loc,
        # 'xf': xf,
        # 'mask': mask if cfg.MASK.MASK else None
        outputs = self.model.track(x_crop, self.anchors_cwh)

        ctr_rpn = self._convert_centerness(outputs['ctr_rpn'])
        asp_rpn = self._convert_aspect(outputs['asp_rpn'])

        ctr = self._convert_centerness(outputs['ctr_rcnn'])
        iou = self._convert_centerness(outputs['iou_rcnn'])
        score = ctr * iou

        import ipdb
        ipdb.set_trace()
        pred_bbox = self._convert_bbox(outputs['loc_rcnn'], outputs['roi_rcnn'])

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[:, 2], pred_bbox[:, 3]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[:, 2]/pred_bbox[:, 3]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        # pscore *= self.window
        # pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
        #     self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        # import ipdb
        # ipdb.set_trace()
        bbox = pred_bbox[best_idx, :]
        # bbox[:2] += cfg.TRACK.INSTANCE_SIZE / 2.0
        # iou = IoU(center2corner(bbox), center2corner(np.transpose(self.anchors)))
        bbox /= scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        # cx, cy, width, height = self._bbox_clip(cx, cy, width,
        #                                         height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        return {
                'bbox': bbox,
                'best_score': best_score,
                'best_idx': best_idx,
                'pscore': pscore,
                'score': score,
                'ctr_rcnn': ctr,
                'iou_rcnn': iou,
                'pred_bbox': pred_bbox,
                'ctr_rpn': ctr_rpn,
                'asp_rpn': asp_rpn
               }

