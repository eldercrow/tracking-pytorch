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


def crop_img(prefix, anns, set_crop_base_path, set_img_base_path):
    '''
    '''
    frame_crop_base_path = join(set_crop_base_path, prefix)
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

    im = cv2.imread('{}/{}'.format(set_img_base_path, prefix+'.jpg'))
    hh, ww = im.shape[:2]
    im_shape = np.array([ww, hh, ww, hh], dtype=np.float32)
    bbox_dict = {}

    # avg_chans = np.mean(im, axis=(0, 1))
    for trackid, rect in enumerate(anns):
        if rect[1] <= rect[0] or rect[3] <= rect[2]:
            continue
        bbox = np.array([rect[0], rect[2], rect[1], rect[3]])
        bbox *= im_shape
        bbox = np.round(bbox)

        trackkey = '{:02d}'.format(trackid)
        if trackkey not in bbox_dict:
            bbox_dict[trackkey] = {}

        try:
            c_img, cbb = crop_img_bb(im, bbox)
            fn_res = join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, trackid))
            cv2.imwrite(fn_res, c_img)
            bbox_dict[trackkey]['{:06d}'.format(0)] = cbb
        except:
            logging.warning('Could not process {}:{}'.format(frame_crop_base_path, trackkey))
            continue

    fn_json = '{}.json'.format(frame_crop_base_path)
    with open(fn_json, 'w') as fh:
        print(json.dumps({prefix: bbox_dict}, indent=4), file=fh)


def main(path_root, path_res):
    '''
    '''
    crop_path = path_res #os.path.join(path_res, 'crop') #.format(instanc_size)
    if not isdir(crop_path): os.makedirs(crop_path)

    for db_type in ['validation',]:
        set_crop_base_path = join(crop_path, db_type)
        set_img_base_path = join(path_root, db_type, 'images')
    
        fn_csv = join(path_root, db_type, '{}-annotations-bbox.csv'.format(db_type))
        csv_list = open(fn_csv, 'r').read().splitlines()[1:]
    
        # gather per-image gts
        gt_all = ddict(list)
        for one_line in csv_list:
            v = one_line.split(',')
            if int(v[10]) == 1 or int(v[11]) == 1: # no group, depiction
                continue
            gt_all[v[0]].append([float(vi) for vi in v[4:8]])
    
        n_imgs = len(gt_all)
    
        # for k, v in gt_all.items():
        #     crop_img(k, v, set_crop_base_path, set_img_base_path)
    
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_img, k, v,
                                  set_crop_base_path, set_img_base_path) for k, v in gt_all.items()]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_imgs, prefix=db_type, suffix='Done ', barLength=40)
    print('done')

    for db_type in ['validation',]:
        set_crop_base_path = join(crop_path, db_type)
        all_dict = {}
        json_list = glob.glob(os.path.join(set_crop_base_path, '*.json'))
        for fn_json in json_list:
            with open(fn_json, 'r') as fh:
                clip_dict = json.load(fh)
            for k, v in clip_dict.items():
                all_dict['{}/{}'.format(db_type, k)] = v

        with open(os.path.join(crop_path, '{}.json'.format(db_type)), 'w') as fh:
            print(json.dumps(all_dict, indent=4), file=fh)

    for db_type in ['validation',]:
        set_crop_base_path = join(crop_path, db_type)
        json_list = glob.glob(os.path.join(set_crop_base_path, '*.json'))
        for fn_json in json_list:
            os.remove(fn_json)


if __name__ == '__main__':
    path_root = '/home/hyunjoon/dataset/openimage'
    path_res = '/home/hyunjoon/dataset/openimage/crop'

    main(path_root, path_res)
