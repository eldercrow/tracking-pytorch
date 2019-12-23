import os, sys
from os.path import join, isdir
from os import listdir, mkdir, makedirs
import cv2
import json
import numpy as np
import logging
import glob
import xml.etree.ElementTree as ET
from concurrent import futures

from crop_img_bb import crop_img_bb


VID_base_path = '/home/hyunjoon/dataset_jinwook/VID/ILSVRC2015'
ann_base_path = join(VID_base_path, 'Annotations/VID/train/')
sub_sets = sorted({'a', 'b', 'c', 'd', 'e'})
crop_base_path = '/home/hyunjoon/dataset/vid/crop'
num_threads = 20


# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def process_clip(sub_set, video, crop_path):
    '''
    '''
    video_crop_base_path = join(crop_path, sub_set, video)
    if not isdir(video_crop_base_path): makedirs(video_crop_base_path)

    sub_set_base_path = join(ann_base_path, sub_set)
    xmls = sorted(glob.glob(join(sub_set_base_path, video, '*.xml')))

    bbox_dict = {}

    for xml in xmls:
        xmltree = ET.parse(xml)
        # size = xmltree.findall('size')[0]
        # frame_sz = [int(it.text) for it in size]
        objects = xmltree.findall('object')
        objs = []
        filename = xmltree.findall('filename')[0].text

        im = cv2.imread(xml.replace('xml', 'JPEG').replace('Annotations', 'Data'))
        avg_chans = np.mean(im, axis=(0, 1))
        for object_iter in objects:
            trackid = int(object_iter.find('trackid').text)
            trackkey = '{:02d}'.format(trackid)
            if trackkey not in bbox_dict:
                bbox_dict[trackkey] = {}

            # name = (object_iter.find('name')).text
            bndbox = object_iter.find('bndbox')
            # occluded = int(object_iter.find('occluded').text)

            bbox = [float(bndbox.find('xmin').text), float(bndbox.find('ymin').text),
                    float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)]
            try:
                c_img, cbb = crop_img_bb(im, bbox)
                fn_res = join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(filename), trackid))
                cv2.imwrite(fn_res, c_img)
                bbox_dict[trackkey]['{:06d}'.format(int(filename))] = cbb
            except:
                logging.warning('Could not process {}'.format(xml))
                continue

    fn_json = os.path.join(crop_path, '{}_{}.json'.format(sub_set, video))
    with open(fn_json, 'w') as fh:
        print(json.dumps({'{}/{}'.format(sub_set, video): bbox_dict}, indent=4), file=fh)


def main():
    if not isdir(crop_base_path): makedirs(crop_base_path)

    for sub_set in sub_sets:
        sub_set_base_path = os.path.join(ann_base_path, sub_set)
        videos = sorted(listdir(sub_set_base_path))
        n_videos = len(videos)

        # for video in videos:
        #     process_clip(sub_set, video, crop_base_path)
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(process_clip, sub_set, video, crop_base_path) for video in videos]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_videos, prefix=sub_set, suffix='Done ', barLength=40)

    # all the intermediate json files
    train_dict = {}
    json_list = glob.glob(os.path.join(crop_base_path, '*_train_*.json'))
    for fn_json in json_list:
        with open(fn_json, 'r') as fh:
            clip_dict = json.load(fh)
        for k, v in clip_dict.items():
            train_dict[k] = v

    with open(os.path.join(crop_base_path, 'train.json'), 'w') as fh:
        print(json.dumps(train_dict, indent=4), file=fh)

    val_dict = {}
    json_list = glob.glob(os.path.join(crop_base_path, '*_val_*.json'))
    for fn_json in json_list:
        with open(fn_json, 'r') as fh:
            clip_dict = json.load(fh)
        for k, v in clip_dict.items():
            val_dict[k] = v

    with open(os.path.join(crop_base_path, 'val.json'), 'w') as fh:
        print(json.dumps(val_dict, indent=4), file=fh)

    json_list = glob.glob(os.path.join(crop_base_path, '*_train_*.json'))
    for fn_json in json_list:
        os.remove(fn_json)

    json_list = glob.glob(os.path.join(crop_base_path, '*_val_*.json'))
    for fn_json in json_list:
        os.remove(fn_json)


if __name__ == '__main__':
    main()

