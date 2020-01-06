# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch

from pysot.config import cfg
from pysot.utils.anchor import Anchors
from pysot.utils.bbox import IoU, center2corner
from pysot.tracker.base_tracker import SiameseTracker


class SiamCARTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamCARTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHORLESS.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        # self.window = np.power(np.tile(window.flatten(), self.anchor_num), cfg.TRACK.WINDOW_INFLUENCE)
        self.centers = self.generate_center(self.score_size)
        self.model = model
        self.model.eval()

        # self.long_short = long_short

    def generate_center(self, score_size):
        X, Y = np.meshgrid(np.arange(score_size), np.arange(score_size))

        # center of grid is 0
        centers = np.stack([X, Y], axis=0).astype(np.float32) + 0.5 - score_size / 2.0
        centers *= cfg.ANCHORLESS.STRIDE

        centers = np.reshape(centers, (2, -1))

        return centers

    def _convert_bbox(self, delta, center):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta = (delta + cfg.ANCHORLESS.OFFSET) * cfg.ANCHORLESS.SCALE

        cx = center[0] + (delta[2] - delta[0])
        cy = center[1] + (delta[3] - delta[1])
        w = delta[0] + delta[2]
        h = delta[1] + delta[3]
        return np.stack([cx, cy, w, h], axis=0)

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _convert_centerness(self, centerness):
        centerness = centerness.permute(1, 2, 3, 0).contiguous().view(-1)
        centerness = torch.sigmoid(centerness).detach().cpu().numpy()
        return centerness

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
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
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
        self.model.template(z_crop)
        zf = self.model.zf

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
        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.centers)
        centerness = self._convert_centerness(outputs['ctr'])

        score *= centerness

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        # pscore *= self.window
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        # import ipdb
        # ipdb.set_trace()
        bbox = pred_bbox[:, best_idx]
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
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

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
               }

