import os, sys
from os.path import join, isdir
from os import listdir, mkdir, makedirs
import cv2
import json
import numpy as np
import logging
import glob
import pickle
import xml.etree.ElementTree as ET
from concurrent import futures
from pycocotools.coco import COCO

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


def crop_img(img, anns, set_crop_base_path, set_img_base_path):
    '''
    '''
    prefix = img['file_name'].split('/')[-1].split('.')[0]
    frame_crop_base_path = join(set_crop_base_path, prefix)
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)

    im = cv2.imread('{}/{}'.format(set_img_base_path, img['file_name']))
    bbox_dict = {}

    # 2019.11.19 new annotation dict with all the objects
    annot_dict = {}

    bbox_all = np.array([ann['bbox'] for ann in anns if ann['iscrowd'] == 0])
    bbox_all[:, 2:] += bbox_all[:, :2]

    # avg_chans = np.mean(im, axis=(0, 1))
    for trackid, ann in enumerate(anns):
        rect = ann['bbox']
        iscrowd = ann['iscrowd']
        bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
        if rect[2] <= 0 or rect[3] <=0 or iscrowd:
            continue

        trackkey = '{:02d}'.format(trackid)
        if trackkey not in bbox_dict:
            bbox_dict[trackkey] = {}
            annot_dict[trackkey] = {}

        try:
            c_img, cbb, cbb_all = crop_img_bb(im, bbox, bbox_all)
            fn_res = join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, trackid))
            cv2.imwrite(fn_res, c_img)
            bbox_dict[trackkey]['{:06d}'.format(0)] = cbb
            annot_dict[trackkey]['{:06d}'.format(0)] = {'gt': cbb, 'objs': cbb_all}
        except:
            logging.warning('Could not process {}:{}'.format(frame_crop_base_path, trackkey))
            continue

    fn_json = '{}.json'.format(frame_crop_base_path)
    with open(fn_json, 'w') as fh:
        print(json.dumps({prefix: bbox_dict}, indent=4), file=fh)

    fn_pkl = '{}.pkl'.format(frame_crop_base_path)
    with open(fn_pkl, 'wb') as fh:
        pickle.dump({prefix: annot_dict}, fh)


def main(path_root, path_res):
    '''
    '''
    dataDir = path_root
    crop_path = path_res #os.path.join(path_res, 'crop') #.format(instanc_size)
    if not isdir(crop_path): os.makedirs(crop_path)

    for dataType in ['val2017', 'train2017']:
        set_crop_base_path = join(crop_path, dataType)
        set_img_base_path = join(dataDir, dataType)

        annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
        coco = COCO(annFile)
        n_imgs = len(coco.imgs)
        # for id in coco.imgs:
        #     crop_img(coco.loadImgs(id)[0],
        #              coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None)),
        #              set_crop_base_path, set_img_base_path)

        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_img, coco.loadImgs(id)[0],
                                  coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None)),
                                  set_crop_base_path, set_img_base_path) for id in coco.imgs]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_imgs, prefix=dataType, suffix='Done ', barLength=40)
    print('done')

    for dataType in ['val2017', 'train2017']:
        set_crop_base_path = join(crop_path, dataType)
        all_dict = {}
        # json_list = glob.glob(os.path.join(set_crop_base_path, '*.json'))
        # for fn_json in json_list:
        #     with open(fn_json, 'r') as fh:
        #         clip_dict = json.load(fh)
        #     for k, v in clip_dict.items():
        #         all_dict['{}/{}'.format(dataType, k)] = v

        # with open(os.path.join(crop_path, '{}.json'.format(dataType)), 'w') as fh:
        #     print(json.dumps(all_dict, indent=4), file=fh)
        pkl_list = glob.glob(os.path.join(set_crop_base_path, '*.pkl'))
        for fn_pkl in pkl_list:
            with open(fn_pkl, 'rb') as fh:
                clip_dict = pickle.load(fh)
            for k, v in clip_dict.items():
                all_dict['{}/{}'.format(dataType, k)] = v

        with open(os.path.join(crop_path, '{}.pkl'.format(dataType)), 'wb') as fh:
            pickle.dump(all_dict, file=fh)

    for dataType in ['val2017', 'train2017']:
        set_crop_base_path = join(crop_path, dataType)
        json_list = glob.glob(os.path.join(set_crop_base_path, '*.json'))
        for fn_json in json_list:
            os.remove(fn_json)
        pkl_list = glob.glob(os.path.join(set_crop_base_path, '*.pkl'))
        for fn_pkl in pkl_list:
            os.remove(fn_pkl)


if __name__ == '__main__':
    path_root = '/home/hyunjoon/dataset_jinwook/COCO'
    path_res = '/home/hyunjoon/dataset/coco/crop'

    main(path_root, path_res)
