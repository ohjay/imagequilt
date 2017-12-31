#!/usr/bin/env python

"""
_transfer_naive.py

Image quilting as per Efros et al.:
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-siggraph01.pdf

Texture transfer.
"""

import os
import argparse
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.color import rgb2gray
from performance import timed

ITR_SCALE = 0.65
CHECKPOINT = True

#####################
# UTILITY FUNCTIONS #
#####################

def ssd(img_patch, template):
    """Computes the SSD between the patch and the template (over all nonzero template regions).
    Assumes that IMG_PATCH and TEMPLATE are the same shape.
    """
    return np.sum(((img_patch - template) ** 2)[template != 0])

def select_patch(img, img_out, pos_y, pos_x, patch_height, patch_width,
                 overlap_height, overlap_width, err_threshold, transfer_info=None):
    """Selects a patch from IMG to be placed in IMG_OUT with the top left corner at (POS_Y, POS_X).
    Returns the patch as a (y, x) tuple representing the position of its top left corner in IMG.
    """
    img_gray, target_gray, alpha, itr = transfer_info
    img_out_above = img_out[pos_y - overlap_height:pos_y, pos_x:pos_x + patch_width]
    ioa_width = img_out_above.shape[1]
    img_out_left = img_out[pos_y:pos_y + patch_height, pos_x - overlap_width:pos_x]
    iol_height = img_out_left.shape[0]
    target_patch = target_gray[pos_y:pos_y + patch_height, pos_x:pos_x + patch_width]
    tp_height, tp_width = target_patch.shape
    existing_patch = img_out[pos_y:pos_y + patch_height, pos_x:pos_x + patch_width]
    ep_height, ep_width, _ = existing_patch.shape

    err_map = np.empty((img_height, img_width))
    err_map.fill(np.inf)
    for y in range(overlap_height, img_height - patch_height):
        for x in range(overlap_width, img_width - patch_width):
            err_map[y, x] = 0
            # Overlap error
            overlap_mult = 0.5 * alpha if pos_y > 0 and pos_x > 0 else alpha
            if itr > 0:
                overlap_mult *= 0.5
            if pos_y > 0:
                err_map[y, x] += overlap_mult * ssd(img[y - overlap_height:y, x:x + ioa_width], img_out_above)
            if pos_x > 0:
                err_map[y, x] += overlap_mult * ssd(img[y:y + iol_height, x - overlap_width:x], img_out_left)
            # Target error
            err_map[y, x] += (1 - alpha) * ssd(img_gray[y:y + tp_height, x:x + tp_width], target_patch)
            # Error with existing patch
            if itr > 0:
                err_map[y, x] += 0.5 * alpha * ssd(img[y:y + ep_height, x:x + ep_width], existing_patch)

    p_choices = np.where(err_map <= np.min(err_map) * (1.0 + err_threshold))
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
                py, px = select_patch(img, img_out, y, x, patch_height, patch_width, overlap_height, overlap_width,
                                      err_threshold, transfer_info=(img_gray, target_gray, alpha, itr))
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
            skio.imsave(outpath[:-4] + '_itr%d.jpg' % (itr + 1), img_out)
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
    parser.add_argument('--err_threshold',  '-tol', type=float, default=0.05)
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
        _min_dim = min(img_height, img_width)
        args.patchsize = max(1, min(_min_dim // 2, out_width // 10))
    if not args.patch_height:
        args.patch_height = args.patchsize
    if not args.patch_width:
        args.patch_width = args.patchsize

    if not args.overlap:
        args.overlap = max(1, args.patchsize // 3)
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
