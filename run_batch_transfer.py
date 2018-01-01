#!/usr/bin/env python

"""
run_batch_transfer.py

Transfer textures from a batch of samples to a batch of targets.

Usage:
    run_batch_transfer.py <sample0> <target0> <sample1> <target1> ...
"""

import os
import argparse
import numpy as np
import skimage as sk
import skimage.io as skio
from main import transfer

def grouped(iterable, n):
    """Source: https://stackoverflow.com/a/5389547."""
    return zip(*[iter(iterable)] * n)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepaths', nargs='*')
    args = parser.parse_args()

    for texture_path, target_path in grouped(args.filepaths, 2):
        print('[o] Transferring texture from `%s` to `%s`.' % (texture_path, target_path))
        img = skio.imread(texture_path)
        img = sk.img_as_float(img).astype(np.float32)
        img_height, img_width, nc = img.shape
        patch_height = max(1, min(img_height, img_width) // 3)
        patch_width = max(1, min(img_height, img_width) // 3)
        overlap_height = max(1, patch_height // 3)
        overlap_width = max(1, patch_width // 3)
        target_img = skio.imread(target_path)
        target_img = sk.img_as_float(target_img).astype(np.float32)
        texture_base = texture_path[texture_path.rfind('/') + 1:texture_path.rfind('.')]
        target_base = target_path[target_path.rfind('/') + 1:target_path.rfind('.')]
        outpath = os.path.join('out', texture_base + '_' + target_base + '.jpg')
        transfer(img, target_img, patch_height, patch_width,
                 overlap_height, overlap_width, 0.0, 0.1, 7, outpath)
