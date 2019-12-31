# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.models.non_local.non_local import APNB, MultiAPNB

NONLOCAL = {
            'NonLocal': APNB,
            'MultiNonLocal': MultiAPNB
           }

def get_nonlocal(name, **kwargs):
    return NONLOCAL[name](**kwargs)
