import cv2
import numpy as np
import copy
import itertools

from dataflow.dataflow import imgaug, BatchData, MultiProcessMapDataZMQ, MapData
# from tensorpack.utils import logger
from pysot.datasets.dataset import TrkDataset
from pysot.datasets.augmentation import (
        ShiftScaleAugmentor, ResizeAugmentor,
        ColorJitterAugmentor, GrayscaleAugmentor, MotionBlurAugmentor,
        box_to_point8, point8_to_box)
from pysot.datasets.anchor_target import AnchorTarget
from pysot.config import cfg

from torch.utils.data import IterableDataset


class TPIterableDataset(IterableDataset):
    '''
    '''
    def __init__(self, df):
        super(TPIterableDataset, self).__init__()
        self._dataflow = df
        self._dataflow.reset_state()

    def __iter__(self):
        return self._dataflow.__iter__()

    def __len__(self):
        return self._dataflow.__len__()


# def aspect_weight(bbox, ratios):
#     '''
#     '''
#     asp = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
#     asp = np.log(asp)
#     log_ratios = np.log(ratios)
#
#     # upper
#     try:
#         uidx = np.min(np.where(log_ratios > asp)[0])
#     except:
#         uidx = len(ratios) - 1
#     try:
#         bidx = np.max(np.where(log_ratios <= asp)[0])
#     except:
#         bidx = 0
#
#     assert (uidx - bidx) >= 0 and (uidx - bidx) <= 1
#
#     weights = np.zeros((len(ratios)), dtype=np.float32)
#     if uidx == bidx:
#         weights[uidx] = 1.0
#     else:
#         r = log_ratios[uidx] - log_ratios[bidx]
#         wu = (asp - log_ratios[bidx]) / r
#         wb = 1 - wu
#         weights[uidx] = wu
#         weights[bidx] = wb
#     return weights


class MalformedData(BaseException):
    pass


class TrainingDataPreprocessor:
    """
    The mapper to preprocess the input data for training.
    Since the mapping may run in other processes, we write a new class and
    explicitly pass cfg to it, in the spirit of "explicitly pass resources to subprocess".
    """
    def __init__(self, cfg):
        self.cfg = cfg
        # augmentations:
        #   shift, scale, blur, flip, grayscale
        template_augmentors = [
            ColorJitterAugmentor(),
            ShiftScaleAugmentor(
                self.cfg.DATASET.TEMPLATE.PAD_RATIO,
                self.cfg.DATASET.TEMPLATE.SHIFT,
                self.cfg.DATASET.TEMPLATE.SCALE,
                self.cfg.DATASET.TEMPLATE.ASPECT,
                self.cfg.TRAIN.EXEMPLAR_SIZE,
                self.cfg.PREPROC.PIXEL_MEAN[::-1]),
            # ResizeAugmentor(self.cfg.TRAIN.EXEMPLAR_SIZE),
        ]
        if self.cfg.DATASET.TEMPLATE.BLUR:
            template_augmentors.append(
                MotionBlurAugmentor(self.cfg.DATASET.TEMPLATE.BLUR)
            )
        # if cfg.DATASET.GRAY:
        #     template_augmentors.append(GrayscaleAugmentor(cfg.DATASET.GRAY))
        self.template_aug = imgaug.AugmentorList(template_augmentors)

        search_augmentors = [
            ColorJitterAugmentor(),
            ShiftScaleAugmentor(
                self.cfg.DATASET.SEARCH.PAD_RATIO,
                self.cfg.DATASET.SEARCH.SHIFT,
                self.cfg.DATASET.SEARCH.SCALE,
                self.cfg.DATASET.SEARCH.ASPECT,
                self.cfg.TRAIN.SEARCH_SIZE,
                self.cfg.PREPROC.PIXEL_MEAN[::-1]),
            # ResizeAugmentor(self.cfg.TRAIN.SEARCH_SIZE),
        ]
        if self.cfg.DATASET.SEARCH.BLUR:
            search_augmentors.append(
                MotionBlurAugmentor(self.cfg.DATASET.SEARCH.BLUR)
            )
        # if cfg.DATASET.GRAY:
        #     template_augmentors.append(GrayscaleAugmentor(cfg.DATASET.GRAY))
        self.search_aug = imgaug.AugmentorList(search_augmentors)

        self.anchor_target = AnchorTarget()

    def __call__(self, datum_dict):
        '''
        datum_dict: dict of {'template', 'search', 'neg'}
        '''
        template, search, is_neg = datum_dict['template'], datum_dict['search'], datum_dict['neg']

        # load images
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        # boxes are in the range of [x0, x1), [y0, y1)
        template_box = template[1] #self._get_bbox(template_image, template[1])
        search_box = search[1] #self._get_bbox(search_image, search[1])
        self.template_aug.augmentors[1].set_bbox(template_box)
        self.search_aug.augmentors[1].set_bbox(search_box)

        # augmentation
        # random transforms
        tfms_t = self.template_aug.get_transform(template_image)
        tfms_s = self.search_aug.get_transform(search_image)

        # flip augmentation should be entangled
        if np.random.uniform() < 0.5:
            tsz = self.cfg.TRAIN.EXEMPLAR_SIZE
            ssz = self.cfg.TRAIN.SEARCH_SIZE
            tfms_t += imgaug.FlipTransform(tsz, tsz, horiz=True)
            tfms_s += imgaug.FlipTransform(ssz, ssz, horiz=True)

        # apply transforms
        template_image = tfms_t.apply_image(template_image)
        points = box_to_point8(template_box).astype(np.float32)
        points = tfms_t.apply_coords(points)
        template_box = point8_to_box(points)
        # template_box = np.reshape(template_box, [-1, 4])

        # w_anchor = aspect_weight(template_box, cfg.ANCHOR.RATIOS)

        search_image = tfms_s.apply_image(search_image)
        points = box_to_point8(search_box).astype(np.float32)
        points = tfms_s.apply_coords(points)
        search_box = point8_to_box(points)

        # centering
        hh, ww = search_image.shape[:2]
        search_box = np.array(search_box, dtype=np.float32)
        # search_box -= np.array([ww/2.0, hh/2.0, ww/2.0, hh/2.0], dtype=np.float32)
        # search_box = np.reshape(search_box, [-1, 4])

        # # get scaled bounding box
        # template_box = self._get_bbox(template_image, template[1])
        # search_box = self._get_bbox(search_image, search[1])

        # anchor target setting - here or in the network?
        cls, delta, delta_weight, overlap, x_pos = self.anchor_target(
                search_box, cfg.TRAIN.OUTPUT_SIZE, is_neg)

        # finally, augment x_pos
        rx, ry = np.random.randint(-2, 3, 2)
        x_pos[0::2] += rx
        x_pos[1::2] += ry

        ret = { \
                'template': np.transpose(template_image, (2, 0, 1)).astype(np.float32),
                'search': np.transpose(search_image, (2, 0, 1)).astype(np.float32),
                'label_cls': cls,
                'label_loc': delta,
                'label_loc_weight': delta_weight,
                'bbox': search_box,
                'x_pos': x_pos,
                }
                # 'template_box': template_box
        return ret


def get_train_dataflow():
    '''
    training dataflow with data augmentation.
    '''
    ds = TrkDataset()
    train_preproc = TrainingDataPreprocessor(cfg)

    if cfg.TRAIN.NUM_WORKERS == 1:
        ds = MapData(ds, train_preproc)
    else:
        ds = MultiProcessMapDataZMQ(ds, cfg.TRAIN.NUM_WORKERS, train_preproc)
    ds = BatchData(ds, cfg.TRAIN.BATCH_SIZE)
    return TPIterableDataset(ds)

