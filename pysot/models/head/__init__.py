# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.head.rpn import UPChannelRPN, DepthwiseRPN, MultiRPN, MaxPoolRPN, MultiMaxPoolRPN

RPNS = {
        'UPChannelRPN': UPChannelRPN,
        'DepthwiseRPN': DepthwiseRPN,
        'MultiRPN': MultiRPN,
        'MultiMaxPoolRPN': MultiMaxPoolRPN,
        'MaxPoolRPN': MaxPoolRPN,
       }


def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)
