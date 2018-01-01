#!/usr/bin/env python

"""
run_batch_synthesis.py

Synthesize textures from a batch of samples.

Usage:
    run_batch_synthesis.py [--outsize <int>] <filepath0> <filepath1> <filepath2> ...
"""

import os
import argparse
import numpy as np
import skimage as sk
import skimage.io as skio
from main import synthesis

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outsize', type=int)
    parser.add_argument('texture_paths', nargs='*')
    args = parser.parse_args()

    for texture_path in args.texture_paths:
        print('[o] Synthesizing texture from `%s`.' % texture_path)
        img = skio.imread(texture_path)
        img = sk.img_as_float(img).astype(np.float32)
        img_height, img_width, nc = img.shape
        patch_height = max(1, min(img_height, img_width) // 3)
        patch_width = max(1, min(img_height, img_width) // 3)
        overlap_height = max(1, patch_height // 3)
        overlap_width = max(1, patch_width // 3)
        out_height = args.outsize if args.outsize else max(img_height, img_width) * 3
        out_width = args.outsize if args.outsize else max(img_height, img_width) * 3
        err_threshold = 0.15
        texture_base = texture_path[texture_path.rfind('/') + 1:texture_path.rfind('.')]
        outpath = os.path.join('out', texture_base + '.jpg')
        synthesis(img, out_height, out_width, patch_height, patch_width,
                  overlap_height, overlap_width, err_threshold, outpath)
