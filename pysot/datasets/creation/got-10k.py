import os, sys
import cv2
import json
import numpy as np
import logging
from concurrent import futures

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


def process_clip(path_root, res_root, prefix, clip_str):
    '''
    '''
    path_clip = os.path.join(path_root, prefix, clip_str)
    path_res = os.path.join(res_root, prefix, clip_str)

    if not os.path.exists(path_res):
        os.makedirs(path_res)

    gt_str = open(os.path.join(path_clip, 'groundtruth.txt'), 'r').read().splitlines()
    nframe = len(gt_str)

    bbox_dict = {}
    for ii, gt in enumerate(gt_str, 1):
        bbox = [float(b) for b in gt.split(',')]
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        fn_img = os.path.join(path_clip, '{:08d}.jpg'.format(ii))
        img = cv2.imread(fn_img)
        assert img is not None

        try:
            c_img, cbb = crop_img_bb(img, bbox)
            fn_res = os.path.join(path_res, '{:08d}.00.x.jpg'.format(ii))
            cv2.imwrite(fn_res, c_img)
            bbox_dict['{:08d}'.format(ii)] = cbb
        except:
            logging.warning('Could not process {}'.format(fn_img))
            continue

    res_dict = {'00': bbox_dict}
    fn_json = os.path.join(res_root, prefix, '{}.json'.format(clip_str))
    with open(fn_json, 'w') as fh:
        print(json.dumps({'{}/{}'.format(prefix, clip_str): res_dict}, indent=4), file=fh)


def create_db(path_root, res_root, prefix, num_threads=20):
    '''
    prefix: 'train' or 'val'
    '''
    # all the clips
    clips_str = open(os.path.join(path_root, prefix, 'list.txt'), 'r').read().splitlines()
    n_clips = len(clips_str)

    pp = os.path.join(res_root, prefix)
    if not os.path.exists(pp):
        os.makedirs(pp)

    gt_dict = {}

    # remain_clips_str = []
    # for clip_str in clips_str:
    #     fn_json = os.path.join(res_root, prefix, '{}.json'.format(clip_str))
    #     if not os.path.exists(fn_json):
    #         remain_clips_str.append(clip_str)

    # for clip_str in clips_str:
    #     process_clip(path_root, res_root, prefix, clip_str)

    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(process_clip, path_root, res_root, prefix, clip_str)
                for clip_str in clips_str]
        for i, f in enumerate(futures.as_completed(fs)):
            printProgress(i, len(clips_str), prefix=prefix, suffix='Done ', barLength=40)

    for clip_str in clips_str:
        fn_json = os.path.join(res_root, prefix, '{}.json'.format(clip_str))
        with open(fn_json, 'r') as fh:
            clip_dict = json.load(fh)
        for k, v in clip_dict.items():
            gt_dict[k] = v

    #     path_clip = os.path.join(path_root, prefix, clip_str)
    #     path_res = os.path.join(res_root, prefix, clip_str)

    #     if not os.path.exists(path_res):
    #         os.makedirs(path_res)

    #     clip_dict = process_clip(path_root, res_root, prefix, clip_str)
    #     gt_dict['{}/{}'.format(prefix, clip_str)] = clip_dict

    with open(os.path.join(res_root, '{}.json'.format(prefix)), 'w') as fh:
        print(json.dumps(gt_dict, indent=4), file=fh)

    for clip_str in clips_str:
        fn_json = os.path.join(res_root, prefix, '{}.json'.format(clip_str))
        os.remove(fn_json)

def main(path_root, path_res):
    # prefix = 'train'
    create_db(path_root, path_res, 'val')
    create_db(path_root, path_res, 'train')


if __name__ == '__main__':
    path_root = '/home/hyunjoon/dataset/got-10k'
    path_res = '/home/hyunjoon/dataset/got-10k/crop'

    main(path_root, path_res)
