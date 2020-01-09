# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.head.rpn import DepthwiseRPN
from pysot.models.head.rcnn import DepthwiseRCNN

RPNS = {
        'DepthwiseRPN': DepthwiseRPN,
       }

def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)


RCNNS = {
        'DepthwiseRCNN': DepthwiseRCNN,
       }

def get_rcnn_head(name, **kwargs):
    return RCNNS[name](**kwargs)