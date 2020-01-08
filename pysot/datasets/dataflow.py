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
# from pysot.datasets.anchorless_target import AnchorlessTarget
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
            ShiftScaleAugmentor(
                self.cfg.DATASET.TEMPLATE.PAD_RATIO,
                self.cfg.DATASET.TEMPLATE.SHIFT,
                self.cfg.DATASET.TEMPLATE.SCALE,
                self.cfg.DATASET.TEMPLATE.ASPECT,
                self.cfg.TRAIN.EXEMPLAR_SIZE,
                self.cfg.PREPROC.PIXEL_MEAN[::-1]),
            # ResizeAugmentor(self.cfg.TRAIN.EXEMPLAR_SIZE),
            ColorJitterAugmentor(),
        ]
        if self.cfg.DATASET.TEMPLATE.BLUR:
            template_augmentors.append(
                MotionBlurAugmentor(self.cfg.DATASET.TEMPLATE.BLUR)
            )
        # if cfg.DATASET.GRAY:
        #     template_augmentors.append(GrayscaleAugmentor(cfg.DATASET.GRAY))
        self.template_aug = imgaug.AugmentorList(template_augmentors)

        search_augmentors = [
            ShiftScaleAugmentor(
                self.cfg.DATASET.SEARCH.PAD_RATIO,
                self.cfg.DATASET.SEARCH.SHIFT,
                self.cfg.DATASET.SEARCH.SCALE,
                self.cfg.DATASET.SEARCH.ASPECT,
                self.cfg.TRAIN.SEARCH_SIZE,
                self.cfg.PREPROC.PIXEL_MEAN[::-1]),
            # ResizeAugmentor(self.cfg.TRAIN.SEARCH_SIZE),
            ColorJitterAugmentor(),
        ]
        if self.cfg.DATASET.SEARCH.BLUR:
            search_augmentors.append(
                MotionBlurAugmentor(self.cfg.DATASET.SEARCH.BLUR)
            )
        # if cfg.DATASET.GRAY:
        #     template_augmentors.append(GrayscaleAugmentor(cfg.DATASET.GRAY))
        self.search_aug = imgaug.AugmentorList(search_augmentors)

        self.anchor_target = AnchorTarget()
        # self.anchorless_target = AnchorlessTarget()

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
        self.template_aug.augmentors[0].set_bbox(template_box)
        self.search_aug.augmentors[0].set_bbox(search_box)

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

        search_image = tfms_s.apply_image(search_image)
        points = box_to_point8(search_box).astype(np.float32)
        points = tfms_s.apply_coords(points)
        search_box = point8_to_box(points)

        cls, centerness, anchor_2nd, iou_2nd, ctr_2nd, delta_2nd, delta_w_2nd = \
                self.anchor_target(search_box, cfg.TRAIN.SEARCH_SIZE, is_neg)

        ret = { \
                'template': np.transpose(template_image, (2, 0, 1)).astype(np.float32),
                'search': np.transpose(search_image, (2, 0, 1)).astype(np.float32),
                'template_box': template_box,
                'search_box': search_box,
                'label_cls': cls,
                'label_ctr': centerness,
                'anchor_2nd': anchor_2nd,
                'label_iou_2nd': iou_2nd,
                'label_ctr_2nd': ctr_2nd,
                'label_delta_2nd': delta_2nd,
                'label_delta_w_2nd': delta_w_2nd,
                }
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

