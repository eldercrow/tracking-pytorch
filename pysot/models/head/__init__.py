# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.head.rpn import DepthwiseRPN
from pysot.models.head.car_head import CARHead
from pysot.models.head.corr_head import CorrHead

RPNS = {
        'DepthwiseRPN': DepthwiseRPN,
        'CARHead': CARHead,
        'CorrHead': CorrHead
       }


def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)
