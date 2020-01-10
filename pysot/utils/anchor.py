# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import numpy as np

from pysot.utils.bbox import corner2center, center2corner


def generate_anchor(out_shape, scales, ratios, stride):
    '''
    return:
        anchors: [4, s*r, out_shape, out_shape], or
                 [4, out_shape, out_shape] if s*r is 1.
        anchors_cwh: [4, s*r, out_shape, out_shape], or
                     [4, out_shape, out_shape] if s*r is 1.
        centers: [2, out_shape, out_shape]
    '''
    X, Y = np.meshgrid(np.arange(out_shape),
                       np.arange(out_shape))

    centers = np.stack([X, Y], axis=0).astype(np.float32)
    centers += (0.5 - out_shape / 2.0)
    cx, cy = centers[0], centers[1]
    twos = np.ones_like(cx) * 2.0

    anchors = []
    anchors_cwh = []
    for s in scales:
        for r in ratios:
            sx2 = s * np.sqrt(r) / 2.0
            sy2 = s / np.sqrt(r) / 2.0

            x0 = cx - sx2
            y0 = cy - sy2
            x1 = cx + sx2
            y1 = cy + sy2

            anchors.append(np.stack([x0, y0, x1, y1], axis=0))
            anchors_cwh.append(np.stack([cx, cy, sx2*twos, sy2*twos], axis=0))
    anchors = np.transpose(np.stack(anchors, axis=0), (1, 0, 2, 3))
    anchors = np.squeeze(anchors)
    anchors_cwh = np.transpose(np.stack(anchors_cwh, axis=0), (1, 0, 2, 3))
    anchors_cwh = np.squeeze(anchors_cwh)

    # centers *= stride
    anchors *= stride
    anchors_cwh *= stride
    return anchors, anchors_cwh #, centers


class Anchors:
    """
    This class generate anchors.
    """
    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = 0
        self.size = 0

        self.anchor_num = len(self.scales) * len(self.ratios)

        self.anchors = None

        self.generate_anchors()

    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size*1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w*0.5, -h*0.5, w*0.5, h*0.5][:]
                count += 1

    def generate_all_anchors(self, im_c, size):
        """
        im_c: image center
        size: image size
        """
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size

        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1),
                             [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                            np.stack([cx, cy, w, h]).astype(np.float32))
        return True
