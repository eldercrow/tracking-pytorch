import os, sys
import cv2
import json
import numpy as np
import logging
from concurrent import futures

from collections import ordereddict as odict

from crop_img_bb import crop_img_bb


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


def _parse_metadata(fn_gt, path_root):
    '''
    '''
    path_gt = os.path.split(fn_gt)[0]
    path_img = os.path.join(path_gt, 'img')
    fn_occ = os.path.join(path_gt, 'full_occlusion.txt')
    fn_oob = os.path.join(path_gt, 'out_of_view.txt')

    gt_all = open(os.path.join(path_root, fn_gt), 'r').read().splitlines()
    occ_all = open(os.path.join(path_root, fn_occ), 'r').read().split(',')
    oob_all = open(os.path.join(path_root, fn_oob), 'r').read().split(',')

    metadata = []
    for ii, (gt, occ, oob) in enumerate(zip(gt_all, occ_all, oob_all), 1):
        bbox = [float(g) for g in gt.split(',')]
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        fn_img = os.path.join(path_img, '{:08d}.jpg'.format(ii))

        metadata.append([fn_img, bbox[0], bbox[1], bbox[2], bbox[3], int(occ), int(oob)])
    return path_gt, metadata


def process_clip(path_root, res_root, clip_str):
    '''
    '''
    # prefix: 'train/train-1'
    prefix, metadata = _parse_metadata(clip_str, path_root)
    path_res = os.path.join(res_root, prefix)

    if not os.path.exists(path_res):
        os.makedirs(path_res)

    nframe = len(metadata)

    # import ipdb
    # ipdb.set_trace()
    bbox_dict = {}
    for gt in metadata[::3]:
        bbox = gt[1:5]
        occ = gt[-2]
        oob = gt[-1]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or occ or oob:
            continue
        fn_img = os.path.join(path_root, gt[0])
        img = cv2.imread(fn_img)
        assert img is not None

        ii = os.path.splitext(os.path.split(fn_img)[-1])[0]

        try:
            c_img, cbb = crop_img_bb(img, bbox)
            fn_res = os.path.join(path_res, '{}.00.x.jpg'.format(ii))
            cv2.imwrite(fn_res, c_img)
            bbox_dict['{}'.format(ii)] = cbb
        except:
            logging.warning('Could not process {}'.format(fn_img))
            continue

    res_dict = {'00': bbox_dict}
    fn_json = os.path.join(res_root, '{}.json'.format(prefix))
    with open(fn_json, 'w') as fh:
        print(json.dumps({'{}'.format(prefix): res_dict}, indent=4), file=fh)


def create_db(path_root, res_root, num_threads=20):
    '''
    '''
    gt_all = open(os.path.join(path_root, 'gt_all.txt'), 'r').read().splitlines()
    n_clips = len(gt_all)

    if not os.path.exists(res_root):
        os.makedirs(res_root)

    gt_dict = {}

    # remain_clips_str = []
    # for clip_str in clips_str:
    #     fn_json = os.path.join(res_root, prefix, '{}.json'.format(clip_str))
    #     if not os.path.exists(fn_json):
    #         remain_clips_str.append(clip_str)

    # for clip_str in gt_all:
    #     process_clip(path_root, res_root, clip_str)

    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(process_clip, path_root, res_root, fn_gt)
                for fn_gt in gt_all]
        for i, f in enumerate(futures.as_completed(fs)):
            printProgress(i, n_clips, prefix='lasot', suffix='Done ', barLength=40)

    for fn_gt in gt_all:
        fn_json = os.path.join(res_root, '{}.json'.format(os.path.split(fn_gt)[0]))
        with open(fn_json, 'r') as fh:
            clip_dict = json.load(fh)
        for k, v in clip_dict.items():
            gt_dict[k] = v

    #     path_clip = os.path.join(path_root, prefix, fn_gt)
    #     path_res = os.path.join(res_root, prefix, fn_gt)

    #     if not os.path.exists(path_res):
    #         os.makedirs(path_res)

    #     clip_dict = process_clip(path_root, res_root, prefix, fn_gt)
    #     gt_dict['{}/{}'.format(prefix, fn_gt)] = clip_dict

    with open(os.path.join(res_root, 'lasot.json'), 'w') as fh:
        print(json.dumps(gt_dict, indent=4), file=fh)

    for fn_gt in gt_all:
        fn_json = os.path.join(res_root, '{}.json'.format(os.path.split(fn_gt)[0]))
        os.remove(fn_json)

def main(path_root, path_res):
    create_db(path_root, path_res)


if __name__ == '__main__':
    path_root = '/home/hyunjoon/dataset/lasot/lasot'
    path_res = '/home/hyunjoon/dataset/lasot/crop'

    main(path_root, path_res)
