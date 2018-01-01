#!/usr/bin/env python

"""
main.py

Image quilting as per Efros et al.:
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-siggraph01.pdf

Texture synthesis and transfer.

Usage:
    main.py
    [--texture <texture_path>] [--target <image_path>]
    [--out_height <int>] [--out_width <int>] [--outsize <int>]
    [--patch_height <int>] [--patch_width <int>] [--patchsize <int>]
    [--overlap_height <int>] [--overlap_width <int>] [--overlap <int>]
    [--err_threshold <float>] [--alpha_init <float>] [--n <int>] [--high_fidelity] [--outdir <str>]

e.g.
    main.py --texture in/bricks_small.jpg --outsize 576 --patchsize 32 --overlap 16
"""

import os
import sys
import argparse
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.util import view_as_windows
from performance import timed
try:
    import cv2
except ImportError:
    pass

DEFAULT_TEXTURE = 'in/sn_small.jpg'
DOUBLE_OVERLAP = False
ITR_SCALE = 0.65
HIGH_FIDELITY = False

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

def similarity(img, template, method='cv2'):
    """Returns the similarity map for all relevant positions in IMG.
    To this end there are multiple methods / similarity metrics:
    - ssd (similarity for each overlap computed as -SSD)
    - cv2 (similarity for each overlap computed as OpenCV magic)

    In this function one of these methods will automatically be selected and applied.
    """
    if method == 'cv2' and 'cv2' in sys.modules:
        return similarity_cv2(img, template)
    _err_map = error_ssd(img, template)
    return np.max(_err_map) - _err_map

def error(img, template, method='cv2'):
    """Returns the error map for all relevant positions in IMG."""
    if method == 'cv2' and 'cv2' in sys.modules:
        _simi_map = similarity_cv2(img, template)
        return np.max(_simi_map) - _simi_map
    return error_ssd(img, template)

def process_similarity_map(simi_map, err_threshold):
    """Takes a similarity map and an error threshold
    and returns only the indices in SIMI_MAP whose values are below the threshold.
    If there are no such indices, returns the indices corresponding to the maximum value in SIMI_MAP.
    Indices will be returned as a list of (y, x) tuples.
    """
    _min, _max = np.min(simi_map), np.max(simi_map)
    p_choices = np.where(simi_map >= _min + (1.0 - err_threshold) * (_max - _min))
    if len(p_choices[0]) == 0:
        return [np.unravel_index(np.argmax(simi_map), simi_map.shape)]
    return zip(*p_choices)

def select_patch_synthesis(img, img_out, pos_y, pos_x, patch_height, patch_width,
                           oh_subtrahend, oh_addend, ow_subtrahend, ow_addend, err_threshold):
    """Selects a patch from IMG to be placed in IMG_OUT with the top left corner at (POS_Y, POS_X).
    Returns the patch as a (y, x) tuple representing the position of its top left corner in IMG.
    """
    if (pos_y, pos_x) == (0, 0):
        sel_y = int(np.random.random_sample() * (img_height - patch_height))
        sel_x = int(np.random.random_sample() * (img_width - patch_width))
        return sel_y, sel_x
    u_choices = set()
    if pos_y > 0:
        template = img_out[pos_y - oh_subtrahend:pos_y + oh_addend, pos_x:pos_x + patch_width]
        _img = img[:-patch_height - oh_subtrahend - oh_addend, ow_subtrahend:-patch_width - ow_addend]
        simi_map = similarity(_img, template, method='cv2')
        u_choices = set(process_similarity_map(simi_map, err_threshold))
    l_choices = set()
    if pos_x > 0:
        template = img_out[pos_y:pos_y + patch_height, pos_x - ow_subtrahend:pos_x + ow_addend]
        _img = img[oh_subtrahend:-patch_height - oh_addend, :-patch_width - ow_subtrahend - ow_addend]
        simi_map = similarity(_img, template, method='cv2')
        l_choices = set(process_similarity_map(simi_map, err_threshold))
    choices = u_choices & l_choices
    if len(choices) == 0:
        choices = u_choices | l_choices
    y, x = tuple(choices)[np.random.randint(len(choices))]
    return y + oh_subtrahend, x + ow_subtrahend

def select_patch_transfer(img, img_out, pos_y, pos_x, patch_height, patch_width,
                          overlap_height, overlap_width, err_threshold, target_img, alpha, itr):
    """Selects a patch from IMG to be placed in IMG_OUT with the top left corner at (POS_Y, POS_X).
    Returns the patch as a (y, x) tuple representing the position of its top left corner in IMG.
    """
    img_out_above = img_out[pos_y - overlap_height:pos_y, pos_x:pos_x + patch_width]
    img_out_left = img_out[pos_y:pos_y + patch_height, pos_x - overlap_width:pos_x]
    target_patch = target_img[pos_y:pos_y + patch_height, pos_x:pos_x + patch_width]
    existing_patch = img_out[pos_y:pos_y + patch_height, pos_x:pos_x + patch_width]

    # Define selection range within texture sample
    margin_below, margin_right = img_out_left.shape[0], img_out_above.shape[1]
    y_lower, y_upper = overlap_height, img_height - margin_below + 1
    x_lower, x_upper = overlap_width, img_width - margin_right + 1

    err_map = np.empty((img_height, img_width))
    err_map.fill(np.inf)
    # Target error
    _img = img[overlap_height:, overlap_width:]
    target_mult = 1.0 if HIGH_FIDELITY else 1.0 - alpha
    err_map[y_lower:y_upper, x_lower:x_upper] = target_mult * error(_img, target_patch)
    # Local texture error
    overlap_mult = 0.5 * alpha if pos_y > 0 and pos_x > 0 else alpha
    if itr > 0:
        overlap_mult *= 0.5
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

def synthesis(img, out_height, out_width, patch_height, patch_width,
              overlap_height, overlap_width, err_threshold, outpath):
    img_out = np.zeros((out_height, out_width, nc)).astype(np.float32)
    if DOUBLE_OVERLAP:
        oh_subtrahend, oh_addend = overlap_height // 2, overlap_height - overlap_height // 2
        ow_subtrahend, ow_addend = overlap_width // 2, overlap_width - overlap_width // 2
    else:
        oh_subtrahend, oh_addend = overlap_height, 0
        ow_subtrahend, ow_addend = overlap_width, 0
    for y in range(0, out_height, patch_height):
        for x in range(0, out_width, patch_width):
            py, px = select_patch_synthesis(img, img_out, y, x, patch_height, patch_width,
                                            oh_subtrahend, oh_addend, ow_subtrahend, ow_addend, err_threshold)
            dy, dx, _ = img_out[y:y + patch_height + oh_addend, x:x + patch_width + ow_addend].shape  # manage overflow
            img_out[y:y + dy, x:x + dx] = img[py:py + dy, px:px + dx]
            if y > 0:
                # Horizontal cut
                y_patch = img[py - oh_subtrahend:py + oh_addend, px:px + dx]
                y_overlapped = img_out[y - oh_subtrahend:y + oh_addend, x:x + dx]
                ypt, yot = y_patch.transpose(1, 0, 2), y_overlapped.transpose(1, 0, 2)
                img_out[y - oh_subtrahend:y + oh_addend, x:x + dx] = vcut(ypt, yot).transpose(1, 0, 2)
            if x > 0:
                # Vertical cut
                x_patch = img[py:py + dy, px - ow_subtrahend:px + ow_addend]
                x_overlapped = img_out[y:y + dy, x - ow_subtrahend:x + ow_addend]
                img_out[y:y + dy, x - ow_subtrahend:x + ow_addend] = vcut(x_patch, x_overlapped)
    skio.imshow(img_out)
    skio.show()
    skio.imsave(outpath, img_out)
    print('Output saved to %s.' % outpath)

def transfer(img, target_img, patch_height, patch_width,
             overlap_height, overlap_width, err_threshold, alpha_init, n, outpath):
    out_height, out_width, _ = target_img.shape
    img_out = np.zeros((out_height, out_width, nc)).astype(np.float32)
    alpha = alpha_init
    _scale = lambda v: max(1, int(v * ITR_SCALE))
    for itr in range(n):
        print('[o] Iteration %02d / %02d...' % (itr + 1, n))
        for y in range(0, out_height, patch_height):
            for x in range(0, out_width, patch_width):
                py, px = select_patch_transfer(img, img_out, y, x, patch_height, patch_width,
                                               overlap_height, overlap_width, err_threshold, target_img, alpha, itr)
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
        if itr < n - 1:
            skio.imsave(outpath[:-4] + '_itr%d.jpg' % (itr + 1), img_out)
    skio.imshow(img_out)
    skio.show()
    skio.imsave(outpath, img_out)
    print('Output saved to %s.' % outpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--texture',        '-tex', type=str)
    parser.add_argument('--target',         '-tar', type=str)
    parser.add_argument('--out_height',     '-oh',  type=int)
    parser.add_argument('--out_width',      '-ow',  type=int)
    parser.add_argument('--outsize',        '-os',  type=int)
    parser.add_argument('--patch_height',   '-ph',  type=int)
    parser.add_argument('--patch_width',    '-pw',  type=int)
    parser.add_argument('--patchsize',      '-ps',  type=int)
    parser.add_argument('--overlap_height', '-ovh', type=int)
    parser.add_argument('--overlap_width',  '-ovw', type=int)
    parser.add_argument('--overlap',        '-ov',  type=int)
    parser.add_argument('--err_threshold',  '-tol', type=float)
    parser.add_argument('--alpha_init',     '-a',   type=float, default=0.1)
    parser.add_argument('--n',              '-n',   type=int,   default=8)
    parser.add_argument('--high_fidelity',  '-hf',  action='store_true')
    parser.add_argument('--outdir',         '-out', type=str,   default='out')
    args = parser.parse_args()

    if not args.texture:
        print('No texture provided! Falling back to default texture %s.' % DEFAULT_TEXTURE)
        args.texture = DEFAULT_TEXTURE
    img = skio.imread(args.texture)
    img = sk.img_as_float(img).astype(np.float32)
    img_height, img_width, nc = img.shape
    texture_base = args.texture[args.texture.rfind('/') + 1:args.texture.rfind('.')]

    if not args.patchsize:
        args.patchsize = max(1, min(img_height, img_width) // 3)
    if not args.patch_height:
        args.patch_height = args.patchsize
    if not args.patch_width:
        args.patch_width = args.patchsize
    if not args.overlap_height:
        args.overlap_height = args.overlap if args.overlap else max(1, args.patch_height // 3)
    if not args.overlap_width:
        args.overlap_width = args.overlap if args.overlap else max(1, args.patch_width // 3)
    assert args.overlap_height < args.patch_height, 'overlap must be contained within patch height'
    assert args.overlap_width < args.patch_width,   'overlap must be contained within patch width'

    if args.target:
        # Texture transfer
        target_img = skio.imread(args.target)
        target_img = sk.img_as_float(target_img).astype(np.float32)
        if not args.err_threshold:
            args.err_threshold = 0.05
        if args.high_fidelity:
            HIGH_FIDELITY = True
        target_base = args.target[args.target.rfind('/') + 1:args.target.rfind('.')]
        outpath = os.path.join(args.outdir, texture_base + '_' + target_base + '.jpg')
        transfer(img, target_img, args.patch_height, args.patch_width, args.overlap_height, args.overlap_width,
                 args.err_threshold, args.alpha_init, args.n, outpath)
    else:
        # Texture synthesis
        if not args.outsize:
            args.outsize = max(img_height, img_width) * 3
        if not args.out_height:
            args.out_height = args.outsize
        if not args.out_width:
            args.out_width = args.outsize
        if not args.err_threshold:
            args.err_threshold = 0.15
        outpath = os.path.join(args.outdir, texture_base + '.jpg')
        synthesis(img, args.out_height, args.out_width, args.patch_height, args.patch_width,
                  args.overlap_height, args.overlap_width, args.err_threshold, outpath)
