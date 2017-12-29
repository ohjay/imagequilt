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
    'patchsize':     60,
    'overlap':       20,
    'err_threshold': 0.15,
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

def select_patch(img, img_out, pos_y, pos_x,
                 patch_height, patch_width, overlap_height, overlap_width, err_threshold):
    """Selects a patch from IMG to be placed in IMG_OUT with the top left corner at (POS_Y, POS_X).
    Returns the patch as a (y, x) tuple representing the position of its top left corner in IMG.
    """
    img_height, img_width, nc = img.shape
    if (pos_y, pos_x) == (0, 0):
        sel_y = int(np.random.random_sample() * (img_height - patch_height))
        sel_x = int(np.random.random_sample() * (img_width - patch_width))
        return sel_y, sel_x
    u_choices = set()
    if pos_y > 0:
        template = img_out[pos_y-overlap_height:pos_y, pos_x:pos_x + patch_width]
        simi_map = similarity(img[:-patch_height - overlap_height, overlap_width:-patch_width], template, method='cv2')
        u_choices = set([(y + overlap_height, x + overlap_width)
                         for y, x in process_similarity_map(simi_map, err_threshold)])
    l_choices = set()
    if pos_x > 0:
        template = img_out[pos_y:pos_y + patch_height, pos_x - overlap_width:pos_x]
        simi_map = similarity(img[overlap_height:-patch_height, :-patch_width - overlap_width], template, method='cv2')
        l_choices = set([(y + overlap_width, x + overlap_width)
                         for y, x in process_similarity_map(simi_map, err_threshold)])
    choices = u_choices & l_choices
    if len(choices) == 0:
        choices = u_choices | l_choices
    return tuple(choices)[np.random.randint(len(choices))]

def vcut(patch, overlapped):
    """Calculates the minimum error VERTICAL boundary cut through the overlapped region.
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

def synthesize(texture_path, out_height, out_width, patch_height, patch_width,
               overlap_height, overlap_width, err_threshold, outdir):
    img = skio.imread(texture_path)
    img = sk.img_as_float(img).astype(np.float32)
    img_height, img_width, nc = img.shape
    img_out = np.zeros((out_height, out_width, nc)).astype(np.float32)
    for y in range(0, out_height, patch_height):
        for x in range(0, out_width, patch_width):
            py, px = select_patch(img, img_out, y, x, patch_height, patch_width,
                                  overlap_height, overlap_width, err_threshold)
            dy, dx, _ = img_out[y:y + patch_height, x:x + patch_width].shape  # account for overflow
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

    assert args.overlap_height < args.patch_height, 'overlap must be contained within patch height'
    assert args.overlap_width < args.patch_width,   'overlap must be contained within patch width'

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
