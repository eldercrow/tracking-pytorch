# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamrpn_r50_l234_dwxcorr"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
# Positive anchor threshold
__C.TRAIN.THR_HIGH = 0.6
# Negative anchor threshold
__C.TRAIN.THR_LOW = 0.3

# Number of negative
__C.TRAIN.NEG_NUM = 16
# Number of positive
__C.TRAIN.POS_NUM = 16
# Number of anchors per images
__C.TRAIN.TOTAL_NUM = 64

__C.TRAIN.NUM_ROI = 32

__C.TRAIN.EXEMPLAR_SIZE = 127
__C.TRAIN.SEARCH_SIZE = 255
__C.TRAIN.BASE_SIZE = 8
__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''
__C.TRAIN.PRETRAINED = ''
__C.TRAIN.LOG_DIR = './logs'
__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20
__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0
__C.TRAIN.LOC_WEIGHT = 1.2
__C.TRAIN.CTR_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 100
__C.TRAIN.LOG_GRADS = False
__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005
__C.TRAIN.LR = CN()
__C.TRAIN.LR.TYPE = 'log'
__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()
__C.TRAIN.LR_WARMUP.WARMUP = True
__C.TRAIN.LR_WARMUP.TYPE = 'step'
__C.TRAIN.LR_WARMUP.EPOCH = 5
__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.PREPROC = CN()
__C.PREPROC.PIXEL_MEAN = [123.675, 116.28, 103.53]
__C.PREPROC.PIXEL_STD = [58.395, 57.12, 57.375]

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
__C.DATASET.TEMPLATE = CN()
__C.DATASET.TEMPLATE.PAD_RATIO = 2
__C.DATASET.TEMPLATE.SHIFT = 4.0 / 64.0 #4
__C.DATASET.TEMPLATE.SCALE = 1.1111
__C.DATASET.TEMPLATE.ASPECT = 1.1
__C.DATASET.TEMPLATE.BLUR = 0.0
__C.DATASET.TEMPLATE.FLIP = 0.0
__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()
__C.DATASET.SEARCH.PAD_RATIO = 4
__C.DATASET.SEARCH.SHIFT = 1.0 #64
__C.DATASET.SEARCH.SCALE = 1.25
__C.DATASET.SEARCH.ASPECT = 1.2
__C.DATASET.SEARCH.BLUR = 0.0
__C.DATASET.SEARCH.FLIP = 0.0
__C.DATASET.SEARCH.COLOR = 1.0

# __C.DATASET.TEMPLATE = CN()
# # Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# # for detail discussion
# __C.DATASET.TEMPLATE.SHIFT = 4
# __C.DATASET.TEMPLATE.SCALE = 0.05
# __C.DATASET.TEMPLATE.BLUR = 0.0
# __C.DATASET.TEMPLATE.FLIP = 0.0
# __C.DATASET.TEMPLATE.COLOR = 1.0
#
# __C.DATASET.SEARCH = CN()
# __C.DATASET.SEARCH.SHIFT = 64
# __C.DATASET.SEARCH.SCALE = 0.18
# __C.DATASET.SEARCH.BLUR = 0.0
# __C.DATASET.SEARCH.FLIP = 0.0
# __C.DATASET.SEARCH.COLOR = 1.0

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
__C.DATASET.NEG = 0.2

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('GOT10K', 'VID', 'COCO', 'DET', 'YOUTUBEBB')

__C.DATASET.GOT10K = CN()
__C.DATASET.GOT10K.ROOT = '~/dataset/got-10k/crop'
__C.DATASET.GOT10K.ANNO = '~/dataset/got-10k/crop/train.json'
__C.DATASET.GOT10K.FRAME_RANGE = 90
__C.DATASET.GOT10K.NUM_USE = 100000

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = '~/dataset/vid/crop'
__C.DATASET.VID.ANNO = '~/dataset/vid/crop/train.json'
__C.DATASET.VID.FRAME_RANGE = 90
__C.DATASET.VID.MIN_FRAME_RANGE = 1
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = 'training_dataset/yt_bb/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = '~/dataset/coco/crop'
__C.DATASET.COCO.ANNO = '~/dataset/coco/crop/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.MIN_FRAME_RANGE = 0
__C.DATASET.COCO.NUM_USE = 100000

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = '~/dataset/imagenet_det/crop'
__C.DATASET.DET.ANNO = '~/dataset/imagenet_det/crop/imagenet_det.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = -1

__C.DATASET.LASOT = CN()
__C.DATASET.LASOT.ROOT = '~/dataset/lasot/crop'
__C.DATASET.LASOT.ANNO = '~/dataset/lasot/crop/lasot.json'
__C.DATASET.LASOT.FRAME_RANGE = 30
__C.DATASET.LASOT.MIN_FRAME_RANGE = 1
__C.DATASET.LASOT.NUM_USE = 200000

__C.DATASET.OPENIMAGE = CN()
__C.DATASET.OPENIMAGE.ROOT = '~/dataset/openimage/crop'
__C.DATASET.OPENIMAGE.ANNO = '~/dataset/openimage/crop/train.json'
__C.DATASET.OPENIMAGE.FRAME_RANGE = 1
__C.DATASET.OPENIMAGE.MIN_FRAME_RANGE = 0
__C.DATASET.OPENIMAGE.NUM_USE = 200000

# NOT USED
__C.DATASET.VIDEOS_PER_EPOCH = 600000

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'resnet18'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True
__C.ADJUST.CROP_SIZE = 7

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# Non-local layer options
# ------------------------------------------------------------------------ #
__C.NONLOCAL = CN()

__C.NONLOCAL.TYPE = 'MultiNonLocal'
__C.NONLOCAL.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.RPN = CN()

# RPN type
__C.RPN.TYPE = 'MultiRPN'

__C.RPN.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Anchor options
# ------------------------------------------------------------------------ #
__C.ANCHOR = CN()

# Anchor stride
__C.ANCHOR.STRIDE = 8

# Anchor ratios
__C.ANCHOR.RATIOS = [0.333, 0.5, 1, 2, 3]

# Anchor scales
__C.ANCHOR.SCALES = [8]

# Anchor number
__C.ANCHOR.ANCHOR_NUM = len(__C.ANCHOR.RATIOS) * len(__C.ANCHOR.SCALES)


# ------------------------------------------------------------------------ #
# Anchorless options
# ------------------------------------------------------------------------ #
__C.RCNN = CN()

__C.RCNN.TYPE = 'DepthwiseRCNN'

__C.RCNN.NUM_ROI = 8

__C.RCNN.KWARGS = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamRPNTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 255

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5
