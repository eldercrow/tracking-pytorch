# import cv2
import numpy as np
import os, sys
import json
# from termcolor import colored
# from tabulate import tabulate

# import tensorflow as tf
import logging
# from tensorpack.utils import logger
from dataflow import RNGDataFlow
# from tensorpack.utils.timer import timed_operation
# from tensorpack.utils.argtools import log_once

from pysot.config import cfg


class SubDataset(object):
    '''
    '''
    def __init__(self, name, root, anno, frame_range, min_frame_range=0):
        '''
        '''
        self.name = name
        self.root = root
        self.anno = anno
        self.frame_range = frame_range
        self.min_frame_range = min_frame_range

        logger = logging.getLogger()
        # load metadata from .json annotation file
        logger.info("loading {} from {}".format(name, anno))
        with open(anno, 'r') as fh:
            meta_data = json.load(fh)
            meta_data = self._filter_zero(meta_data)

        # load annotations
        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self._videos = list(meta_data.keys())
        self.indices = [v for v in self._videos]
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'

    def __len__(self):
        return len(self.indices)

    def shuffle_resize(self, size, shuffle=True):
        '''
        '''
        if size < 0: # no repeat
            if shuffle:
                pick = np.random.permutation(self._videos).tolist()
            else:
                pick = [v in self._videos]
            size = len(pick)
        else:
            pick = []
            m = 0
            while m < size:
                if shuffle:
                    pick.extend(np.random.permutation(self._videos).tolist())
                else:
                    pick.extend([v in self._videos])
                m = len(pick)
        self.indices = pick[:size]
        # logger.info("shuffle done!")
        # logger.info("dataset length {}".format(self.num))
        # return pick[:self.num]

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w < 3 or h < 3:
                            # logging.getLogger().warning("{}/{} has too small bbox, skipping.".format(video, frm))
                            continue
                        # if w <= 0 or h <= 0:
                        #     continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def get_image_anno(self, video, track, frame):
        '''
        '''
        try:
            frame_str = "{:06d}".format(frame)
            image_path = os.path.join(self.root, video,
                                      self.path_format.format(frame_str, track, 'x'))
            assert os.path.exists(image_path)
        except AssertionError: # got-10k
            frame_str = "{:08d}".format(frame)
            image_path = os.path.join(self.root, video,
                                      self.path_format.format(frame_str, track, 'x'))
        try:
            image_anno = self.labels[video][track][frame_str]
        except:
            print(image_path)
            raise KeyError
        return image_path, image_anno

    def get_positive_pair(self, video_name):
        '''
        '''
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        #
        # left = max(template_frame - self.frame_range, 0)
        # right = min(template_frame + self.frame_range, len(frames)-1) + 1
        # search_range = frames[left:right]
        #
        if self.frame_range < 0:
            search_range = frames
        else:
            ll = max(template_frame - self.frame_range, 0)
            lr = max(template_frame - self.min_frame_range, 0) + 1
            l_search_range = frames[ll:lr]
            rl = min(template_frame + self.min_frame_range, len(frames)-1)
            rr = min(template_frame + self.frame_range, len(frames)-1) + 1
            r_search_range = frames[rl:rr]
            search_range = l_search_range + r_search_range
        #
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range) if search_range else template_frame
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, video_name=''):
        if not video_name:
            index = np.random.randint(0, self.__len__())
            video_name = self.indices[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)


class TrkDataset(RNGDataFlow):
    '''
    '''
    def __init__(self, shuffle=True):
        '''
        Get and merge all sub datasets
        '''
        self.shuffle = shuffle

        self.all_dataset = []
        # start = 0
        # self.num = 0
        # self.all_nums = []
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    os.path.expanduser(subdata_cfg.ROOT),
                    os.path.expanduser(subdata_cfg.ANNO),
                    subdata_cfg.FRAME_RANGE,
                    # subdata_cfg.MIN_FRAME_RANGE
                )
            sub_dataset.shuffle_resize(subdata_cfg.NUM_USE, shuffle=shuffle)
            # sub_dataset.log()
            self.all_dataset.append(sub_dataset)
            # self.all_nums.append(subdata_cfg.NUM_USE)

    def __len__(self):
        return np.sum([len(d) for d in self.all_dataset])

    def __iter__(self):
        '''
        '''
        # first reset all sub datasets
        for db in self.all_dataset:
            db.shuffle_resize(len(db), shuffle=self.shuffle)

        # merge all
        all_data_list = []
        for i, db in enumerate(self.all_dataset):
            all_data_list.extend([(i, datum) for datum in db.indices])

        if self.shuffle:
            np.random.shuffle(all_data_list)

        # populate one by one
        for (db_idx, datum) in all_data_list:
            neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()
            if neg:
                template = self.all_dataset[db_idx].get_random_target(datum)
                search = np.random.choice(self.all_dataset).get_random_target()
                neg = template[0] != search[0]
            else:
                template, search = self.all_dataset[db_idx].get_positive_pair(datum)
            #
            yield {'template': template, 'search': search, 'neg': int(neg)}
