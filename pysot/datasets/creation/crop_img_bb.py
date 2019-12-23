import numpy as np
import cv2


def _ioa(boxes, canvas):
    ix = np.minimum(boxes[:, 2], canvas[2]) - np.maximum(boxes[:, 0], canvas[0])
    iy = np.minimum(boxes[:, 3], canvas[3]) - np.maximum(boxes[:, 1], canvas[1])
    inter = np.maximum(ix, 0) * np.maximum(iy, 0)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return np.where(area == 0, np.zeros_like(inter), inter / area)


def crop_img_bb(img, bb, bb_all=None, pad_ratio=6):
    '''
    '''
    H, W = img.shape[:2]
    bb = np.maximum(0, np.minimum([W, H, W, H], np.around(bb)))

    ww, hh = bb[2:] - bb[:2]
    cx, cy = (bb[:2] + bb[2:]) * 0.5

    pad = np.sqrt(ww * hh) * pad_ratio

    ix0 = max(0.0, cx - pad * 0.5)
    iy0 = max(0.0, cy - pad * 0.5)
    ix1 = min(float(W), cx + pad * 0.5)
    iy1 = min(float(H), cy + pad * 0.5)

    ix0, iy0, ix1, iy1 = np.around([ix0, iy0, ix1, iy1]).astype(int)

    x0 = bb[0] - ix0
    y0 = bb[1] - iy0

    assert x0 >= 0 and y0 >= 0

    c_img = img[iy0:iy1, ix0:ix1, :]
    bb = np.array([x0, y0, x0+ww, y0+hh])

    if bb_all is not None:
        ioa = _ioa(bb_all, [ix0, iy0, ix1, iy1])
        pidx = np.where(ioa > 0.6)[0]
        rbb_all = bb_all[pidx, :].copy()
        rbb_all[:, 0::2] -= ix0
        rbb_all[:, 1::2] -= iy0
        

    # optional scale
    max_sz = max(c_img.shape)
    if max_sz > 768:
        sf = max_sz / 768.0
        hh, ww = np.around(np.array(c_img.shape[:2]) / sf).astype(int)
        c_img = cv2.resize(c_img, (ww, hh))
        bb /= sf
        if bb_all is not None:
            rbb_all = np.around(rbb_all / sf, 1)
            rbb_all = np.maximum([[0, 0, 0, 0]], np.minimum([[ww, hh, ww, hh]], rbb_all))

    return c_img, np.around(bb, 1).tolist(), rbb_all
