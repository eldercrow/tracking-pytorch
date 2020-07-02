# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from pysot.config import cfg
from pysot.utils.bbox import IoU, corner2center
from pysot.utils.anchor import Anchors


class AnchorlessTarget:
    def __init__(self,):
        X, Y = np.meshgrid(np.arange(cfg.TRAIN.OUTPUT_SIZE),
                           np.arange(cfg.TRAIN.OUTPUT_SIZE))

        self.centers = np.stack([X, Y], axis=0).astype(np.float32) + 0.5 - cfg.TRAIN.OUTPUT_SIZE / 2.0
        self.centers *= cfg.ANCHORLESS.STRIDE

    def __call__(self, target, sz_img, neg=False):
        '''
        target: target bbox in pixel coordinate, [x0, y0, x1, y1]
        sz_img: patch size in pixel coordinate (typically 255)
        '''
        sz_anc = cfg.TRAIN.OUTPUT_SIZE

        # -1 ignore 0 negative 1 positive
        # cls = -1 * np.ones((sz_anc, sz_anc), dtype=np.float32)
        cls = -1 * np.ones((sz_anc, sz_anc), dtype=np.int64)
        delta = np.zeros((4, sz_anc, sz_anc), dtype=np.float32)
        delta_weight = np.zeros((sz_anc, sz_anc), dtype=np.float32)
        centerness = -1 * np.ones((sz_anc, sz_anc), dtype=np.float32)

        # centering target
        target_c = target - sz_img / 2.0

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target_c)
        x0, y0, x1, y1 = target_c

        # l, u, r, b
        delta[0] = self.centers[0] - x0
        delta[1] = self.centers[1] - y0
        delta[2] = x1 - self.centers[0]
        delta[3] = y1 - self.centers[1]

        pos_mask = np.all(delta > cfg.ANCHORLESS.STRIDE, axis=0)
        neg_mask = np.any(delta < -cfg.ANCHORLESS.STRIDE / 2, axis=0)

        delta = np.maximum(delta, 0)

        if neg:
            cls[pos_mask] = 0
            centerness[pos_mask] = 0
            return cls, delta, delta_weight, centerness

        Mdx = np.maximum(delta[0], delta[2])
        mdx = np.minimum(delta[0], delta[2])
        Mdy = np.maximum(delta[1], delta[3])
        mdy = np.minimum(delta[1], delta[3])
        ctr = np.sqrt((mdx / (Mdx + 1e-08)) * (mdy / (Mdy + 1e-08)))

        # scale-shift delta at here, not before
        delta = delta / cfg.ANCHORLESS.SCALE - cfg.ANCHORLESS.OFFSET

        pos = np.where(pos_mask == True)
        neg = np.where(neg_mask == True)

        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

        cls[pos] = 1
        delta_weight[pos] = 1. / (pos_num + 1e-6)
        # delta_weight[pos] = ctr[pos] / (np.sum(ctr[pos]) + 1e-08)
        centerness[pos] = ctr[pos]

        cls[neg] = 0
        centerness[neg] = 0
        return cls, delta, delta_weight, centerness

