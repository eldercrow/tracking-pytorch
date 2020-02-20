# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.backbone.resnet import resnet18, resnet34, resnet50
from pysot.models.backbone.mobilenetv2 import mobilenet_v2
from pysot.models.backbone.gradnet import gradnet

BACKBONES = {
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'mobilenetv2': mobilenet_v2,
              'gradnet': gradnet,
            }


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
