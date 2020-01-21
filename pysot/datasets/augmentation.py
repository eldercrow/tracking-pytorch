import numpy as np
from dataflow.dataflow.imgaug import Transform, ImageAugmentor, PhotometricAugmentor
from dataflow.dataflow.imgaug import ResizeTransform
import cv2


# class FlipTransform(Transform):
#     """
#     Flip the image.
#     """
#     def __init__(self, h, w, horiz=True):
#         """
#         Args:
#             h, w (int):
#             horiz (bool): whether to flip horizontally or vertically.
#         """
#         self._init(locals())
#
#     def apply_image(self, img):
#         if self.horiz:
#             return img[:, ::-1]
#         else:
#             return img[::-1]
#
#     def apply_coords(self, coords):
#         if self.horiz:
#             coords[:, 0] = self.w - coords[:, 0]
#         else:
#             coords[:, 1] = self.h - coords[:, 1]
#         return coords


class AffineTransform(Transform):
    def __init__(self, roi, out_wh, mean_rgbgr):
        super(AffineTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)

        a = (self.out_wh[0]) / (self.roi[2] - self.roi[0])
        b = (self.out_wh[1]) / (self.roi[3] - self.roi[1])
        c = -a * (self.roi[0] - 0.0)
        d = -b * (self.roi[1] - 0.0)
        self.mapping = np.array([[a, 0, c],
                                 [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(img, self.mapping,
                              self.out_wh,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=self.mean_rgbgr)
        return crop

    def apply_coords(self, coords):
        a, b, c, d = self.mapping.ravel()[[0, 4, 2, 5]]
        coords[:, 0] = coords[:, 0] * a + c
        coords[:, 1] = coords[:, 1] * b + d
        return coords


class CropPadTransform(Transform):
    def __init__(self, x0, y0, x1, y1, mean_rgbgr):
        super(CropPadTransform, self).__init__()
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        ww = x1 - x0
        hh = y1 - y0
        self._init(locals())

    def apply_image(self, img):
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)

        p_img = cv2.copyMakeBorder(img,
                                   1, 1, 1, 1,
                                   cv2.BORDER_CONSTANT,
                                   value=self.mean_rgbgr)
        r_img = cv2.getRectSubPix(p_img, (self.ww, self.hh), (self.cx+0.5, self.cy+0.5))
        return r_img

    def apply_coords(self, coords):
        coords[:, 0] -= self.x0 #(self.cx - self.ww / 2.0)
        coords[:, 1] -= self.y0 #(self.cy - self.hh / 2.0)
        return coords


class ShiftScaleAugmentor(ImageAugmentor):
    """ 
    shift scale the bounding box.
    crop image accordinly, centering the augmented bb.
    """

    def __init__(self,
                 pad_ratio,
                 shift_range,
                 scale_range,
                 aspect_exp,
                 out_size,
                 mean_rgbgr=np.array([127, 127, 127])):
        """
        Randomly crop a box of shape (h, w), sampled from [min, max] (both inclusive).
        If max is None, will use the input image shape.

        Args:
            wmin, hmin, wmax, hmax: range to sample shape.
            max_aspect_ratio (float): the upper bound of ``max(w,h)/min(w,h)``.
        """
        # if max_aspect_ratio is None:
        #     max_aspect_ratio = 9999999
        self._init(locals())
        self.bbox = None

    def set_bbox(self, bbox):
        self.bbox = np.array(bbox, dtype=np.float32)

    def get_transform(self, img):
        assert self.bbox is not None
        
        cx, cy = (self.bbox[:2] + self.bbox[2:]) * 0.5
        h, w = self.bbox[2:] - self.bbox[:2]

        sz = np.sqrt(h*w)
        tsz = sz * self.pad_ratio

        # augmentation params
        rval = self.rng.uniform(size=[5])
        no_aug = rval[0]
        rtx, rty, rs, rasp = rval[1:] * 2.0 - 1.0

        tx = rtx * self.shift_range * sz
        ty = rty * self.shift_range * sz
        ss = np.power(self.scale_range, rs)
        asp = np.power(self.aspect_exp, rasp)
        sx = np.sqrt(ss * asp)
        sy = np.sqrt(ss / asp)
        if no_aug < 0.25: # no scale, only translation
            sx, sy, ss = (1, 1, 1)
            # initial crop box: (cx-t2, cy-t2, cx+t2, cy+t2)
            # cx, cy = image centre, t2 = half of target size

        ww = tsz / sx
        hh = tsz / sy

        # augmentation
        cx += (tx / sx)
        cy += (ty / sy)

        roi = [cx - ww/2.0, cy - hh/2.0, cx + ww/2.0, cy + hh/2.0]

        ww = int(np.around(ww))
        hh = int(np.around(hh))
        # x0 = cx - ww2
        # y0 = cy - hh2
        # x1 = cx + ww2
        # y1 = cy + hh2

        # x0, y0, x1, y1 = np.around([x0, y0, x1, y1]).astype(int)
        # invalidate bbox so that the next call must set bbox first
        self.bbox = None

        return AffineTransform(roi, (self.out_size, self.out_size), self.mean_rgbgr)
        # return CropPadTransform(cx, cy, ww, hh, self.mean_rgbgr)


class ResizeAugmentor(ImageAugmentor):
    def __init__(self, size, interp=cv2.INTER_LINEAR):
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        return ResizeTransform(h, w, self.size, self.size, self.interp)
            
    
class ColorJitterAugmentor(PhotometricAugmentor):
    ''' Random color jittering '''
    def __init__(self, \
                 mean_rgbgr=[127.0, 127.0, 127.0], \
                 rand_l=0.05 * 255, \
                 rand_c=0.1, \
                 rand_h=0.02 * 255):
        super(ColorJitterAugmentor, self).__init__()
        if not isinstance(mean_rgbgr, np.ndarray):
            mean_rgbgr = np.array(mean_rgbgr)
        min_rgbgr = -mean_rgbgr
        max_rgbgr = min_rgbgr + 255.0
        self._init(locals())

    def _get_augment_params(self, _):
        return self.rng.uniform(-1.0, 1.0, [8])

    def _augment(self, img, rval):
        old_dtype = img.dtype
        img = img.astype(np.float32)
        rflag = rval[5:] * 0.5 + 0.5
        rflag = (rflag > 0.3333).astype(float)
        rval[0] *= (self.rand_l * rflag[0])
        rval[1] = np.power(1.0 + self.rand_c, rval[3] * rflag[1])
        rval[2:4] *= (self.rand_h * rflag[2])
        rval[4] = -(rval[2] + rval[3])

        for i in range(3):
            add_val = (rval[0] + rval[i+2] - self.mean_rgbgr[i]) * rval[1] + self.mean_rgbgr[i]
            img[:, :, i] = img[:, :, i] * rval[1] + add_val
            # img[:, :, i] = np.maximum(0.0, np.minimum(255.0,
            #     (img[:, :, i] + add_val) * rval[1] + self.mean_rgbgr[i]))
        img = np.clip(img, 0, 255)
        return img.astype(old_dtype)


class GrayscaleAugmentor(PhotometricAugmentor):
    def __init__(self, rand_g):
        super(GrayscaleAugmentor, self).__init__()
        self._init(locals())

    def _get_augment_params(self, _):
        return self.rng.uniform()

    def _augment(self, image, rval):
        if rval < self.rand_g:
            grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return image


class MotionBlurAugmentor(PhotometricAugmentor):
    """ Gaussian blur the image with random window size"""

    def __init__(self, max_r, min_exp=3.0, max_exp=6.0):
        """
        Args:
            max_r (int): max possible kernel radius
        """
        super(MotionBlurAugmentor, self).__init__()
        self._init(locals())

    def _create_blur_kernel(self, r, e, vx, vy):
        X, Y = np.meshgrid(np.arange(-r, r), np.arange(-r, r))
        W = np.sqrt(X*X + Y*Y)
        W[r, r] = 1.0
        K = np.abs(X * vx + Y * vy) / W
        K[r, r] = 1.0
        K = np.power(K, e)
        return K / np.sum(K)

    def _get_augment_params(self, _):
        rval = self.rng.uniform(0, 1, [4])

        no_aug = rval[0] < 0.25
        r = int(rval[1] * self.max_r) + 1
        e = int(rval[2] * (self.max_exp - self.min_exp)) + self.min_exp

        theta = rval[3] * 2.0 * np.pi
        vx = np.cos(theta)
        vy = np.sin(theta)

        return (no_aug, r, e, vx, vy)

    def _augment(self, img, rval):
        no_aug, r, e, vx, vy = rval
        if not no_aug:
            K = self._create_blur_kernel(r, e, vx, vy)
            img = cv2.filter2D(img, -1, K, borderType=cv2.BORDER_REPLICATE)
        return img


def box_to_point8(boxes, offset_rb=0):
    """
    Args:
        boxes: nx4

    Returns:
        (nx4)x2
    """
    b = np.reshape(boxes, [-1, 4])
    # b[:, 2:] -= offset_rb
    b = b[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    b = b.reshape((-1, 2))
    return b


def point8_to_box(points, offset_rb=0):
    """
    Args:
        points: (nx4)x2
    Returns:
        nx4 boxes (x1y1x2y2)
    """
    p = points.reshape((-1, 4, 2))
    minxy = p.min(axis=1)   # nx2
    maxxy = p.max(axis=1)   # nx2
    b = np.concatenate((minxy, maxxy), axis=1)
    # b[:, 2:] += offset_rb
    if b.shape[0] == 1:
        b = np.ravel(b)
    return b
