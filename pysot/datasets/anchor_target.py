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
        self.anchors, self.anchors_cwh = generate_anchor(cfg.TRAIN.OUTPUT_SIZE, \
                                                         cfg.ANCHOR.SCALES, \
                                                         cfg.ANCHOR.RATIOS, \
                                                         cfg.ANCHOR.STRIDE)

    def __call__(self, target_c, is_neg=False):
        '''
        target: (x0, y0, x1, y1), roi
        '''
        anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)

        sz_anc = cfg.TRAIN.OUTPUT_SIZE

        # -1 ignore 0 negative 1 positive
        # cls_rpn = -1 * np.ones((anchor_num, sz_anc, sz_anc), dtype=np.int64)
        asp = (target_c[2] - target_c[0]) / (target_c[3] - target_c[1])
        aspect_rpn = np.log(asp).astype(np.float32)
        ctr_rpn = -1 * np.ones((sz_anc, sz_anc), dtype=np.float32)
        loc_rpn = np.ones((2, sz_anc, sz_anc), dtype=np.float32)

        # for 2nd stage
        # rcnn_delta = np.zeros((4, cfg.RCNN.NUM_ROI), dtype=np.float32)
        # rcnn_iou = -1 * np.ones((cfg.RCNN.NUM_ROI,), dtype=np.float32)
        # rcnn_ctr = -1 * np.ones((cfg.RCNN.NUM_ROI,), dtype=np.float32)

        def select(position, keep_num=16, do_pad=False):
            num = position[0].shape[0]
            if num == keep_num:
                return position, num

            slt = np.arange(num)
            if num < keep_num and do_pad:
                slt = np.hstack([slt, np.random.randint(0, num, keep_num - num)])
            else:
                np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        # ctr_rpn
        x0, y0, x1, y1 = target_c
        delta = np.stack([self.anchors_cwh[0] - x0, \
                          self.anchors_cwh[1] - y0, \
                          x1 - self.anchors_cwh[0], \
                          y1 - self.anchors_cwh[1]], axis=0)

        loc_rpn[0] = (x1 + x0) * 0.5 - self.anchors_cwh[0]
        loc_rpn[1] = (y1 + y0) * 0.5 - self.anchors_cwh[1]
        loc_rpn[0] /= self.anchors_cwh[2]
        loc_rpn[1] /= self.anchors_cwh[3]

        # [25, 25]
        pos_mask = np.all(delta > 0, axis=0)
        # [4, 25, 25]
        delta = np.maximum(delta, 0)

        # do not regress non-positives
        loc_rpn *= pos_mask

        Mdx = np.maximum(delta[0], delta[2])
        mdx = np.minimum(delta[0], delta[2])
        Mdy = np.maximum(delta[1], delta[3])
        mdy = np.minimum(delta[1], delta[3])
        ctr = np.sqrt((mdx / (Mdx + 1e-08)) * (mdy / (Mdy + 1e-08)))

        pos_all = np.where(pos_mask == True)
        neg_all = np.where(pos_mask == False)

        # pos_2nd, _ = select(pos_all, cfg.TRAIN.NUM_ROI, True)
        # neg_2nd, _ = select(neg_all, cfg.TRAIN.NUM_ROI, True)

        pos, _ = select(pos_all, cfg.TRAIN.POS_NUM)
        neg, _ = select(neg_all, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

        # # (4, NUM_ROI*2)
        # pn_2nd = tuple([np.hstack([p, n]) for p, n in zip(pos_2nd, neg_2nd)])
        # anchors_cwh_2nd = np.stack([self.anchors_cwh[i][pn_2nd] for i in range(4)], axis=0)
        # num_roi = pn_2nd[0].size

        # # randomness to anchors_2nd
        # asp2 = np.sqrt(np.exp(aspect_rpn + np.clip(np.random.normal(0.0, 0.333, num_roi), -1.0, 1.0)))
        # anchors_cwh_2nd = np.stack([ \
        #         anchors_cwh_2nd[0], \
        #         anchors_cwh_2nd[1], \
        #         anchors_cwh_2nd[2] * asp2, \
        #         anchors_cwh_2nd[3] / asp2], axis=0).astype(np.float32)

        # w2 = 0.5 * anchors_cwh_2nd[2]
        # h2 = 0.5 * anchors_cwh_2nd[3]
        # anchors_2nd = np.stack([ \
        #         anchors_cwh_2nd[0] - w2, \
        #         anchors_cwh_2nd[1] - h2, \
        #         anchors_cwh_2nd[0] + w2, \
        #         anchors_cwh_2nd[1] + h2], axis=0).astype(np.float32)

        if is_neg:
            ctr_rpn[pos_mask] = 0
            # ctr_2nd = np.zeros((num_roi,), dtype=np.float32)
            # iou_2nd = np.zeros((num_roi,), dtype=np.float32)
            # loc_2nd = np.zeros((4, num_roi), dtype=np.float32)
            return ctr_rpn, aspect_rpn, loc_rpn, np.random.uniform(0.0, 1.0) * ctr #, \
                #    np.transpose(anchors_2nd, (1, 0)), np.transpose(anchors_cwh_2nd, (1, 0)), \
                #    ctr_2nd, iou_2nd, np.transpose(loc_2nd, (1, 0))

        # label for rpn
        ctr_rpn[pos] = ctr[pos]
        ctr_rpn[neg] = 0

        # # label for rcnn
        # ctr_2nd = ctr[pn_2nd] # [NUM_ROI*2]
        # iou_2nd = IoU(target_c, anchors_2nd) # [NUM_ROI*2]
        # loc_2nd = np.stack([delta[0][pn_2nd] / anchors_cwh_2nd[2], \
        #                     delta[1][pn_2nd] / anchors_cwh_2nd[3], \
        #                     delta[2][pn_2nd] / anchors_cwh_2nd[2], \
        #                     delta[3][pn_2nd] / anchors_cwh_2nd[3]], axis=0)

        return ctr_rpn, aspect_rpn, loc_rpn, np.random.uniform(0.0, 0.3) * ctr #, \
            #    np.transpose(anchors_2nd, (1, 0)), np.transpose(anchors_cwh_2nd, (1, 0)), \
            #    ctr_2nd, iou_2nd, np.transpose(loc_2nd, (1, 0))
