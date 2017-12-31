#!/usr/bin/env python

"""
_transfer.py

Image quilting as per Efros et al.:
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-siggraph01.pdf

Texture transfer.
"""

import os
import sys
import argparse
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.color import rgb2gray
from skimage.util import view_as_windows
from performance import timed
try:
    import cv2
except ImportError:
    pass

ITR_SCALE = 0.7
CHECKPOINT = True

#####################
# UTILITY FUNCTIONS #
#####################

def ssd(img_patch, template):
    """Computes the SSD between the patch and the template (over all nonzero template regions).
    Assumes that IMG_PATCH and TEMPLATE are the same shape.
    """
    return np.sum(((img_patch - template) ** 2)[template != 0])

def error_ssd(img, template):
    """Error metric: sum of squared differences with the template."""
    i_h, i_w, _ = img.shape
    t_h, t_w, _ = template.shape
    result = np.zeros((i_h - t_h, i_w - t_w))
    for y in range(i_h - t_h):
        for x in range(i_w - t_w):
            result[y, x] = ssd(img[y:y + t_h, x:x + t_w], template)
    return result

def error_ssd_vectorized(img, template):
    """I vectorized it but it got even slower."""
    img_view = view_as_windows(img, template.shape) * (template != 0)
    return ((img_view - template) ** 2).sum(axis=(2, 3, 4, 5))

def similarity_cv2(img, template):
    """Similarity metric: convolution with the template.
    Does not work for masked templates.
    """
    return cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)

def error(img, template):
    """Returns the error map for all positions in IMG.
    To this end there are multiple error metrics:
    - ssd (error for each overlap computed as the SSD)
    - cv2 (error for each overlap computed as -convolution similarity)

    In this function one of these metrics will automatically be selected and applied.
    """
    if 'cv2' in sys.modules:
        _simi_map = similarity_cv2(img, template)
        return np.max(_simi_map) - _simi_map
    return error_ssd(img, template)

def select_patch(img, img_out, pos_y, pos_x, patch_height, patch_width,
                 overlap_height, overlap_width, err_threshold, transfer_info=None):
    """Selects a patch from IMG to be placed in IMG_OUT with the top left corner at (POS_Y, POS_X).
    Returns the patch as a (y, x) tuple representing the position of its top left corner in IMG.
    """
    img_gray, target_gray, alpha, itr = transfer_info
    img_out_above = img_out[pos_y - overlap_height:pos_y, pos_x:pos_x + patch_width]
    img_out_left = img_out[pos_y:pos_y + patch_height, pos_x - overlap_width:pos_x]
    target_patch = target_gray[pos_y:pos_y + patch_height, pos_x:pos_x + patch_width]
    existing_patch = img_out[pos_y:pos_y + patch_height, pos_x:pos_x + patch_width]

    # Define selection range within texture sample
    margin_below, margin_right = img_out_left.shape[0], img_out_above.shape[1]
    y_lower, y_upper = overlap_height, img_height - margin_below + 1
    x_lower, x_upper = overlap_width, img_width - margin_right + 1

    err_map = np.empty((img_height, img_width))
    err_map.fill(np.inf)
    # Target error
    _img_gray = img_gray[overlap_height:, overlap_width:]
    err_map[y_lower:y_upper, x_lower:x_upper] = (1 - alpha) * error(_img_gray, target_patch)
    # Local texture error
    overlap_mult = 0.5 * alpha if pos_y > 0 and pos_x > 0 else alpha
    if itr > 0:
        overlap_mult *= 0.5
        _img = img[overlap_height:, overlap_width:]
        err_map[y_lower:y_upper, x_lower:x_upper] += 0.5 * alpha * error(_img, existing_patch)
    if pos_y > 0:
        _img = img[:-margin_below, overlap_width:]
        err_map[y_lower:y_upper, x_lower:x_upper] += overlap_mult * error(_img, img_out_above)
    if pos_x > 0:
        _img = img[overlap_height:, :-margin_right]
        err_map[y_lower:y_upper, x_lower:x_upper] += overlap_mult * error(_img, img_out_left)

    _min, _max = np.min(err_map), np.max(err_map[np.isfinite(err_map)])
    p_choices = np.where(err_map <= _min + err_threshold * (_max - _min))
    if len(p_choices[0]) == 0:
        return np.unravel_index(np.argmin(err_map), err_map.shape)
    return np.random.choice(p_choices[0]), np.random.choice(p_choices[1])

def vcut(patch, overlapped):
    """Calculates the minimum error VERTICAL cut through the overlapped region.
    Then (using that information) cuts the patch, joins it with the overlapped image, and returns the amalgamation.
    """
    err_surface = np.sum((patch - overlapped) ** 2, axis=2)
    cum_err_surface = np.copy(err_surface)
    height, width, _ = patch.shape
    for y in range(1, height):
        for x in range(width):
            cum_err_surface[y, x] = err_surface[y, x] + np.min((cum_err_surface[y - 1, max(x - 1, 0)],
                                                                cum_err_surface[y - 1, x],
                                                                cum_err_surface[y - 1, min(x + 1, width - 1)]))
    x = None
    result = np.zeros_like(patch)
    for y in reversed(range(height)):
        if x is None:
            x = np.argmin(cum_err_surface[y])
        else:
            err_options = (
                float('inf') if x - 1 < 0 else cum_err_surface[y, x - 1],
                cum_err_surface[y, x],
                float('inf') if x + 1 >= width else cum_err_surface[y, x + 1],
            )
            x += np.argmin(err_options) - 1
        result[y, :x] = overlapped[y, :x]
        result[y, x:] = patch[y, x:]
    return result

##################
# MAIN FUNCTIONS #
##################

@timed('Texture transfer')
def transfer(img, target_img, patch_height, patch_width,
             overlap_height, overlap_width, err_threshold, alpha_init, n, outpath):
    img_gray = rgb2gray(img).astype(np.float32)
    target_gray = rgb2gray(target_img).astype(np.float32)
    img_out = np.zeros((out_height, out_width, nc)).astype(np.float32)
    alpha = alpha_init
    _scale = lambda v: max(1, int(v * ITR_SCALE))
    for itr in range(n):
        print('[o] Iteration %02d / %02d...' % (itr + 1, n))
        for y in range(0, out_height, patch_height):
            for x in range(0, out_width, patch_width):
                py, px = select_patch(img, img_out, y, x, patch_height, patch_width,
                                      overlap_height, overlap_width, err_threshold,
                                      transfer_info=(img_gray, target_gray, alpha, itr))
                dy, dx, _ = img_out[y:y + patch_height, x:x + patch_width].shape
                img_out[y:y + dy, x:x + dx] = img[py:py + dy, px:px + dx]
                if y > 0:
                    # Horizontal cut
                    y_patch = img[py - overlap_height:py, px:px + dx]
                    y_overlapped = img_out[y - overlap_height:y, x:x + dx]
                    ypt, yot = y_patch.transpose(1, 0, 2), y_overlapped.transpose(1, 0, 2)
                    img_out[y - overlap_height:y, x:x + dx] = vcut(ypt, yot).transpose(1, 0, 2)
                if x > 0:
                    # Vertical cut
                    x_patch = img[py:py + dy, px - overlap_width:px]
                    x_overlapped = img_out[y:y + dy, x - overlap_width:x]
                    img_out[y:y + dy, x - overlap_width:x] = vcut(x_patch, x_overlapped)
        patch_height, patch_width = _scale(patch_height), _scale(patch_width)
        overlap_height, overlap_width = _scale(overlap_height), _scale(overlap_width)
        err_threshold *= 0.5
        alpha = alpha_init + (0.9 - alpha_init) * (itr + 1) / (n - 1)
        if CHECKPOINT and itr < n - 1:
            skio.imsave(outpath[:-4] + '_itr%d.jpg' % itr, img_out)
    skio.imshow(img_out)
    skio.show()
    skio.imsave(outpath, img_out)
    print('Output saved to %s.' % outpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--texture',        '-tex', type=str,   default='in/sn_small.jpg')
    parser.add_argument('--target',         '-tar', type=str,   default='in/owen.jpg')
    parser.add_argument('--patch_height',   '-ph',  type=int)
    parser.add_argument('--patch_width',    '-pw',  type=int)
    parser.add_argument('--patchsize',      '-ps',  type=int)
    parser.add_argument('--overlap_height', '-ovh', type=int)
    parser.add_argument('--overlap_width',  '-ovw', type=int)
    parser.add_argument('--overlap',        '-ov',  type=int)
    parser.add_argument('--err_threshold',  '-tol', type=float, default=0.5)
    parser.add_argument('--alpha_init',     '-a',   type=float, default=0.1)
    parser.add_argument('--n',              '-n',   type=int,   default=8)
    parser.add_argument('--outdir',         '-out', type=str,   default='out')
    args = parser.parse_args()

    img = skio.imread(args.texture)
    img = sk.img_as_float(img).astype(np.float32)
    img_height, img_width, nc = img.shape
    target_img = skio.imread(args.target)
    target_img = sk.img_as_float(target_img).astype(np.float32)
    out_height, out_width, _ = target_img.shape

    if not args.patchsize:
        args.patchsize = out_width // 10
    if not args.patch_height:
        args.patch_height = args.patchsize
    if not args.patch_width:
        args.patch_width = args.patchsize

    if not args.overlap:
        args.overlap = args.patchsize // 3
    if not args.overlap_height:
        args.overlap_height = args.overlap
    if not args.overlap_width:
        args.overlap_width = args.overlap
    assert args.overlap_height < args.patch_height, 'overlap must be contained within patch height'
    assert args.overlap_width < args.patch_width, 'overlap must be contained within patch width'

    texture_base = args.texture[args.texture.rfind('/') + 1:args.texture.rfind('.')]
    target_base = args.target[args.target.rfind('/') + 1:args.target.rfind('.')]
    outpath = os.path.join(args.outdir, texture_base + '_' + target_base + '.jpg')

    transfer(img, target_img, args.patch_height, args.patch_width,
             args.overlap_height, args.overlap_width, args.err_threshold, args.alpha_init, args.n, outpath)
