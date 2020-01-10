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
from pycocotools.coco import COCO

from collections import defaultdict as ddict

from crop_img_bb import crop_img_bb


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


def crop_img(prefix, xml, crop_path):
    '''
    '''
    xmltree = ET.parse(xml)
    objects = xmltree.findall('object')

    frame_crop_base_path = join(crop_path, prefix) #xml.split('/')[-1].split('.')[0])
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

    img_path = xml.replace('xml', 'JPEG').replace('Annotations', 'Data')
    # img_path = os.path.join(sub_set_base_path, img_path)

    im = cv2.imread(img_path)

    bbox_dict = {}
    for trackid, object_iter in enumerate(objects):
        bndbox = object_iter.find('bndbox')
        bbox = [float(bndbox.find('xmin').text), float(bndbox.find('ymin').text),
                float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)]

        trackkey = '{:02d}'.format(trackid)
        if trackkey not in bbox_dict:
            bbox_dict[trackkey] = {}

        try:
            c_img, cbb, _ = crop_img_bb(im, bbox)
            fn_res = join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, trackid))
            cv2.imwrite(fn_res, c_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            bbox_dict[trackkey]['{:06d}'.format(0)] = cbb
        except:
            logging.warning('Could not process {}:{}'.format(frame_crop_base_path, trackkey))
            continue

    return { prefix: bbox_dict }


def main(path_root, path_res):
    '''
    '''
    crop_path = path_res #os.path.join(path_res, 'crop') #.format(instanc_size)
    if not isdir(crop_path): os.makedirs(crop_path)

    # VID_base_path = './ILSVRC'
    ann_base_path = join(path_root, 'Annotations/DET/train/')
    sub_sets = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i')
    # for sub_set in sub_sets:
    #     sub_set_base_path = join(ann_base_path, sub_set)
    #     if 'a' == sub_set:
    #         xmls = sorted(glob.glob(join(sub_set_base_path, '*', '*.xml')))
    #     else:
    #         xmls = sorted(glob.glob(join(sub_set_base_path, '*.xml')))
    #     prefixes = [xml.replace(ann_base_path, '').replace('.xml', '') for xml in xmls]
    #
    #     n_imgs = len(xmls)
    #     # sub_set_crop_path = join(crop_path, sub_set)
    #
    #     res_dict = {}
    #
    #     # for prefix, xml in zip(prefixes, xmls):
    #     #     f = crop_img(prefix, xml, crop_path)
    #     #     res_dict.update(f)
    #
    #     with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
    #         fs = [executor.submit(crop_img, prefix, xml, crop_path) \
    #                 for prefix, xml in zip(prefixes, xmls)]
    #         for i, f in enumerate(futures.as_completed(fs)):
    #             # Write progress to error so that it can be seen
    #             printProgress(i, n_imgs, prefix=sub_set, suffix='Done ', barLength=40)
    #             res_dict.update(f.result())
    #
    #     print('Number of unprocessed bboxes: {}'.format(n_imgs - len(res_dict)))
    #
    #     fn_json = '{}.json'.format(join(crop_path, sub_set))
    #     with open(fn_json, 'w') as fh:
    #         print(json.dumps(res_dict, indent=4), file=fh)

    all_dict = {}
    for sub_set in sub_sets:
        fn_json = '{}.json'.format(join(crop_path, sub_set))
        with open(fn_json, 'r') as fh:
            clip_dict = json.load(fh)
        all_dict.update(clip_dict)

    with open(os.path.join(crop_path, 'imagenet.json'), 'w') as fh:
        print(json.dumps(all_dict, indent=4), file=fh)

    for sub_set in sub_sets:
        fn_json = '{}.json'.format(join(crop_path, sub_set))
        os.remove(fn_json)

    # for db_type in ['validation',]:
    #     set_crop_base_path = join(crop_path, db_type)
    #     all_dict = {}
    #     json_list = glob.glob(os.path.join(set_crop_base_path, '*.json'))
    #     for fn_json in json_list:
    #         with open(fn_json, 'r') as fh:
    #             clip_dict = json.load(fh)
    #         for k, v in clip_dict.items():
    #             all_dict['{}/{}'.format(db_type, k)] = v
    #
    #     with open(os.path.join(crop_path, '{}.json'.format(db_type)), 'w') as fh:
    #         print(json.dumps(all_dict, indent=4), file=fh)
    #
    # for db_type in ['validation',]:
    #     set_crop_base_path = join(crop_path, db_type)
    #     json_list = glob.glob(os.path.join(set_crop_base_path, '*.json'))
    #     for fn_json in json_list:
    #         os.remove(fn_json)


if __name__ == '__main__':
    path_root = '/local/data/det/ILSVRC2015'
    path_res = '/home/hyunjoon/dataset/imagenet_det/crop'

    main(path_root, path_res)
