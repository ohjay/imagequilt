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
    [--err_threshold <float>] [--n <int>] [--outdir <str>]

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

DEFAULTS = {
    'texture':       'in/bricks_small.jpg',
    'target':        'in/owen_small.jpg',
    'outsize':       600,
    'patchsize':     30,
    'overlap':       10,
    'err_threshold': 0.95,
    'n':             8,
    'outdir':        'out',
}

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

@timed('ssd_vectorized')
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
    return error_ssd(img, template) * -1.0

def select_patch(img, img_out, pos_y, pos_x,
                 patch_height, patch_width, overlap_height, overlap_width, err_threshold):
    """Selects a patch from IMG to be placed in IMG_OUT with the top left corner at (POS_Y, POS_X).
    Returns the patch as a (y, x) tuple representing the position of its top left corner in IMG.
    """
    img_height, img_width, nc = img.shape
    out_height, out_width, _ = img_out.shape
    if (pos_y, pos_x) == (0, 0):
        sel_y = int(np.random.random_sample() * (img_height - patch_height))
        sel_x = int(np.random.random_sample() * (img_width - patch_width))
        return sel_y, sel_x
    elif pos_y == 0:
        template = img_out[pos_y:pos_y + patch_height, pos_x:pos_x + overlap_width]
        simi_map = similarity(img, template, method='cv2')
    elif pos_x == 0:
        template = img_out[pos_y:pos_y + overlap_height, pos_x:pos_x + patch_width]
        simi_map = similarity(img, template, method='cv2')
    else:
        template = img_out[pos_y:pos_y + patch_height, pos_x:pos_x + patch_width]
        simi_map = similarity(img[:-patch_height+overlap_height, :-patch_width+overlap_width], template, method='ssd')
    _min, _max = np.min(simi_map), np.max(simi_map)
    pchoices = np.where(simi_map >= _min + (1.0 - err_threshold) * (_max - _min))
    if len(pchoices[0]) == 0:
        return np.unravel_index(np.argmax(simi_map), simi_map.shape)
    return np.random.choice(pchoices[0]), np.random.choice(pchoices[1])

def cut(patch, overlapped, pos_y, pos_x, overlap_height, overlap_width):
    """Calculates the minimum error boundary cut through the overlap region(s).
    Then (using that information) cuts the patch, joins it with the overlapped image, and returns the amalgamation.
    (POS_Y, POS_X) should be the top left corner of OVERLAPPED in the full output image.
    """
    def _vertical_cut(_patch, _overlapped):
        err_surface = np.sum((_patch - _overlapped) ** 2, axis=2)
        cum_err_surface = np.copy(err_surface)
        height, width, _ = _patch.shape
        for y in range(1, height):
            for x in range(width):
                cum_err_surface[y, x] = err_surface[y, x] + np.min((cum_err_surface[y - 1, max(x - 1, 0)],
                                                                    cum_err_surface[y - 1, x],
                                                                    cum_err_surface[y - 1, min(x + 1, width - 1)]))
        x = None
        result = np.zeros_like(_patch)
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
            result[y, :x] = _overlapped[y, :x]
            result[y, x:] = _patch[y, x:]
        result[result == 0] = _patch[result == 0]
        return result
    if pos_x > 0:
        # Vertical cut
        x_patch, x_overlapped = patch[:, :overlap_width], overlapped[:, :overlap_width]
        patch[:, :overlap_width] = _vertical_cut(x_patch, x_overlapped)
    if pos_y > 0:
        # Horizontal cut
        y_patch, y_overlapped = patch[:overlap_height], overlapped[:overlap_height]
        ypt, yot = np.fliplr(y_patch.transpose(1, 0, 2)), np.fliplr(y_overlapped.transpose(1, 0, 2))
        patch[:overlap_height] = np.fliplr(_vertical_cut(ypt, yot)).transpose(1, 0, 2)
    return patch

##################
# MAIN FUNCTIONS #
##################

def synthesize(texture_path, out_height, out_width, patch_height, patch_width,
               overlap_height, overlap_width, err_threshold, outdir):
    img = skio.imread(texture_path)
    img = sk.img_as_float(img).astype(np.float32)
    img_height, img_width, nc = img.shape
    img_out = np.zeros((out_height, out_width, nc)).astype(np.float32)
    for y in range(0, out_height, patch_height - overlap_height):
        for x in range(0, out_width, patch_width - overlap_width):
            py, px = select_patch(img, img_out, y, x, patch_height, patch_width,
                                  overlap_height, overlap_width, err_threshold)
            patch = img[py:py + patch_height, px:px + patch_width]
            _dy, _dx, _ = np.minimum(img_out[y:y + patch_height, x:x + patch_width].shape, patch.shape)
            img_out[y:y + _dy, x:x + _dx] = cut(patch[:_dy, :_dx], img_out[y:y + _dy, x:x + _dx],
                                                y, x, overlap_height, overlap_width)
        print('%03d / %d ...' % (y, out_height))
    skio.imshow(img_out)
    skio.show()
    outpath = os.path.join(outdir, texture_path[texture_path.rfind('/') + 1:texture_path.rfind('.')] + '.jpg')
    skio.imsave(outpath, img_out)
    print('Output saved to %s.' % outpath)

def transfer(texture_path, target_path, patch_height, patch_width,
             overlap_height, overlap_width, err_threshold, n, outdir):
    pass

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
    parser.add_argument('--n',              '-n',   type=int, default=5)
    parser.add_argument('--outdir',         '-out', type=str)
    args = parser.parse_args()

    if not args.texture:
        print('No texture provided! Falling back to default texture %s.' % DEFAULTS['texture'])
        args.texture = DEFAULTS['texture']
    if not args.patchsize:
        args.patchsize = DEFAULTS['patchsize']
    if not args.patch_height:
        args.patch_height = args.patchsize
    if not args.patch_width:
        args.patch_width = args.patchsize
    if not args.overlap:
        args.overlap = DEFAULTS['overlap']
    if not args.overlap_height:
        args.overlap_height = args.overlap
    if not args.overlap_width:
        args.overlap_width = args.overlap
    if not args.err_threshold:
        args.err_threshold = DEFAULTS['err_threshold']
    if not args.outdir:
        args.outdir = DEFAULTS['outdir']

    if args.target:
        # Texture transfer
        if not args.target:
            args.target = DEFAULTS['target']
        if not args.n:
            args.n = DEFAULTS['n']
        transfer(args.texture, args.target, args.patch_height, args.patch_width,
                 args.overlap_height, args.overlap_width, args.err_threshold, args.n, args.outdir)
    else:
        # Texture synthesis
        if not args.outsize:
            args.outsize = DEFAULTS['outsize']
        if not args.out_height:
            args.out_height = args.outsize
        if not args.out_width:
            args.out_width = args.outsize
        synthesize(args.texture, args.out_height, args.out_width, args.patch_height, args.patch_width,
                   args.overlap_height, args.overlap_width, args.err_threshold, args.outdir)
