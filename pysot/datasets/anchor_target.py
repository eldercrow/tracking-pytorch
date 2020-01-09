# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from pysot.config import cfg
from pysot.utils.bbox import IoU, corner2center
from pysot.utils.anchor import Anchors, generate_anchor


class AnchorTarget:
    def __init__(self,):
        self.anchors, self.centers = generate_anchor(cfg.TRAIN.OUTPUT_SIZE, \
                                                     cfg.ANCHOR.SCALES, \
                                                     cfg.ANCHOR.RATIOS, \
                                                     cfg.ANCHOR.STRIDE)

    def __call__(self, target, sz_img, is_neg=False):
        anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)

        sz_anc = cfg.TRAIN.OUTPUT_SIZE

        # -1 ignore 0 negative 1 positive
        # cls_rpn = -1 * np.ones((anchor_num, sz_anc, sz_anc), dtype=np.int64)
        aspect_rpn = (target[2] - target[0]) / (target[3] - target[1])
        aspect_rpn = np.log(aspect_rpn).astype(np.float32)
        aspect_w_rpn = np.zeros((sz_anc, sz_anc), dtype=np.float32)

        ctr_rpn = -1 * np.ones((sz_anc, sz_anc), dtype=np.float32)
        
        # for 2nd stage
        # rcnn_delta = np.zeros((4, cfg.RCNN.NUM_ROI), dtype=np.float32)
        # rcnn_iou = -1 * np.ones((cfg.RCNN.NUM_ROI,), dtype=np.float32)
        # rcnn_ctr = -1 * np.ones((cfg.RCNN.NUM_ROI,), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        # centering target
        target_c = target - sz_img / 2.0
        
        # compute anchor IoU
        overlap = IoU(self.anchors, target_c) # (5, 25, 25)

        # ctr_rpn
        x0, y0, x1, y1 = target_c
        delta = np.stack([self.centers[0] - x0, \
                          self.centers[1] - y0, \
                          x1 - self.centers[0], \
                          y1 - self.centers[1]], axis=0)

        pos_mask = np.all(delta > 0, axis=0)

        delta = np.maximum(delta, 0)

        if is_neg:
            ctr_rpn[pos_mask] = 0
            return ctr_rpn, aspect_rpn, aspect_w_rpn

        Mdx = np.maximum(delta[0], delta[2])
        mdx = np.minimum(delta[0], delta[2])
        Mdy = np.maximum(delta[1], delta[3])
        mdy = np.minimum(delta[1], delta[3])
        ctr = np.sqrt((mdx / (Mdx + 1e-08)) * (mdy / (Mdy + 1e-08)))

        pos = np.where(pos_mask == True)
        neg = np.where(pos_mask == False)

        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

        ctr_rpn[pos] = ctr[pos]
        ctr_rpn[neg] = 0

        aspect_w_rpn[pos] = 1.0 / pos[0].size

        return ctr_rpn, aspect_rpn, aspect_w_rpn

        # # classification target
        # pos = np.where( \
        #     np.logical_or(ctr > cfg.TRAIN.THR_HIGH, ctr == np.max(ctr)) \
        #     )
        # neg = np.where( \
        #     np.logical_and(ctr < cfg.TRAIN.THR_LOW, overlap < np.max(overlap)) \
        #     )

        # pos_2nd, nn = select(pos, cfg.RCNN.NUM_ROI)
        # if nn < cfg.RCNN.NUM_ROI:
        #     ridx = np.random.randint(0, nn, cfg.RCNN.NUM_ROI)
        #     p2 = [p[ridx] for p in pos_2nd]
        #     pos_2nd = tuple(p2)
        # neg_2nd, nn = select(neg, cfg.RCNN.NUM_ROI)
        # if nn < cfg.RCNN.NUM_ROI:
        #     ridx = np.random.randint(0, nn, cfg.RCNN.NUM_ROI)
        #     p2 = [p[ridx] for p in neg_2nd]
        #     neg_2nd = tuple(p2)
        # pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
        # neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

        # # (4, NUM_ROI*2)
        # pn_2nd = tuple([np.hstack([p,n]) for p, n in zip(pos_2nd, neg_2nd)])
        # anchor_2nd = np.stack([self.anchors[0][pn_2nd], \
        #                        self.anchors[1][pn_2nd], \
        #                        self.anchors[2][pn_2nd], \
        #                        self.anchors[3][pn_2nd]], axis=0)

        # if is_neg:
        #     cls_rpn[pos] = 0
        #     ctr_rpn[pos[1:]] = 0
        #     ctr_2nd = np.zeros((cfg.RCNN.NUM_ROI*2,), dtype=np.float32)
        #     cls_2nd = np.zeros((cfg.RCNN.NUM_ROI*2,), dtype=np.int64)

        #     # for roi align
        #     offset = (sz_anc / 2.0 - 0.5) * cfg.ANCHOR.STRIDE
        #     anchor_2nd = np.transpose(anchor_2nd, (1, 0)) + offset
        #     delta_2nd = np.zeros_like(anchor_2nd)
        #     delta_w_2nd = np.zeros((cfg.RCNN.NUM_ROI*2,), dtype=np.float32)

        #     return cls_rpn, ctr_rpn, anchor_2nd, cls_2nd, ctr_2nd, delta_2nd, delta_w_2nd

        # cls_rpn[pos] = 1
        # cls_rpn[neg] = 0
        # ctr_rpn[pos[1:]] = ctr[pos[1:]]
        # ctr_rpn[neg[1:]] = np.maximum(ctr[neg[1:]], 0)

        # # 2nd stage
        # # regression target 
        # # ctr_rpn
        # # iou
        # ctr_2nd = ctr[pn_2nd[1:]]
        # cls_2nd = np.zeros(cfg.RCNN.NUM_ROI*2, dtype=np.int64)
        # cls_2nd[:cfg.RCNN.NUM_ROI] = 1

        # tcx, tcy, tw, th = corner2center(target_c)
        # cx, cy, w, h = corner2center(anchor_2nd)
        # delta_2nd = np.stack([(tcx - cx) / w, \
        #                       (tcy - cy) / h, \
        #                       np.log(tw / w), \
        #                       np.log(th / h)], axis=0)
        
        # # for roi align
        # # scale = (sz_anc+1.0) / sz_anc
        # # offset = (sz_anc / 2.0 - 0.5) * cfg.ANCHOR.STRIDE
        # anchor_2nd = np.transpose(anchor_2nd, (1, 0)) #+ offset
        # # anchor_2nd = np.maximum(np.minimum(anchor_2nd, sz_anc * cfg.ANCHOR.STRIDE), 0)
        # delta_2nd = np.transpose(delta_2nd, (1, 0))
        # delta_w_2nd = cls_2nd / cfg.RCNN.NUM_ROI

        # return cls_rpn, ctr_rpn, anchor_2nd, cls_2nd, ctr_2nd, delta_2nd, delta_w_2nd