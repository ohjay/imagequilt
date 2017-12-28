#!/usr/bin/env python

"""
tests.py

Run tests with `python -m doctest tests.py`.
"""

import numpy as np
from scipy.signal import fftconvolve

def test_flip():
    """
    >>> template, flipped = test_flip()
    >>> template.shape
    (2, 2, 3)
    >>> flipped.shape
    (2, 2, 3)
    >>> round(template[0, 0, 0], 1)
    0.5
    >>> round(template[0, 0, 1], 1)
    0.9
    >>> round(template[0, 0, 2], 1)
    0.1
    >>> round(template[1, 0, 0], 1)
    0.7
    >>> round(template[1, 0, 1], 1)
    0.2
    >>> round(template[1, 0, 2], 1)
    0.4
    >>> all(template[0, 1, :] == 0.0)
    True
    >>> all(template[1, 1, :] == 0.0)
    True
    >>> all(flipped[0, 0, :] == 0.0)
    True
    >>> all(flipped[1, 0, :] == 0.0)
    True
    >>> round(flipped[0, 1, 0], 1)
    0.7
    >>> round(flipped[0, 1, 1], 1)
    0.2
    >>> round(flipped[0, 1, 2], 1)
    0.4
    >>> round(flipped[1, 1, 0], 1)
    0.5
    >>> round(flipped[1, 1, 1], 1)
    0.9
    >>> round(flipped[1, 1, 2], 1)
    0.1
    """
    template_r = np.array([
        [0.5, 0],
        [0.7, 0],
    ])
    template_g = np.array([
        [0.9, 0],
        [0.2, 0],
    ])
    template_b = np.array([
        [0.1, 0],
        [0.4, 0],
    ])
    template = np.dstack([template_r, template_g, template_b])
    return template, np.flipud(np.fliplr(template))

def test_conv2d():
    """
    >>> result = test_conv2d()
    >>> result.shape
    (2, 3)
    >>> round(result[0, 0], 1)
    0.4
    >>> round(result[0, 1], 1)
    0.8
    >>> round(result[0, 2], 1)
    1.2
    >>> round(result[1, 0], 1)
    1.0
    >>> round(result[1, 1], 1)
    1.0
    >>> round(result[1, 2], 1)
    1.0
    """
    img = np.array([
        [0.3, 0.5, 0.7, 0.9],
        [0.1, 0.3, 0.5, 0.7],
        [0.9, 0.7, 0.5, 0.3],
    ])
    template = np.array([
        [1, 0],
        [1, 0],
    ])
    template = np.flipud(np.fliplr(template))
    return fftconvolve(img, template, mode='valid')

def test_conv3d():
    """
    >>> result = test_conv3d()
    >>> result.shape
    (2, 3)
    >>> round(result[0, 0], 2)
    0.74
    >>> round(result[0, 1], 2)
    1.36
    """
    img_r = np.array([
        [0.3, 0.5, 0.7, 0.9],
        [0.1, 0.3, 0.5, 0.7],
        [0.9, 0.7, 0.5, 0.3],
    ])
    img_g = np.array([
        [0.4, 0.6, 0.8, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.1, 0.7, 0.5, 0.9],
    ])
    img_b = np.array([
        [0.2, 0.2, 0.8, 0.8],
        [0.1, 0.6, 0.5, 0.6],
        [0.5, 0.3, 0.4, 0.7],
    ])
    img = np.dstack([img_r, img_g, img_b])
    template_r = np.array([
        [0.5, 0],
        [0.7, 0],
    ])
    template_g = np.array([
        [0.9, 0],
        [0.2, 0],
    ])
    template_b = np.array([
        [0.1, 0],
        [0.4, 0],
    ])
    template = np.dstack([template_r, template_g, template_b])
    template = np.flipud(np.fliplr(template))
    template[:, :, :] = template[:, :, ::-1]
    return np.squeeze(fftconvolve(img, template, mode='valid'))

def test_transpose():
    """
    >>> z, zt = test_transpose()
    >>> z[0, 0]
    array([1, 5])
    >>> z[0, 1]
    array([2, 6])
    >>> z[1, 0]
    array([3, 7])
    >>> z[1, 1]
    array([4, 8])
    >>> zt[0, 0]
    array([1, 5])
    >>> zt[0, 1]
    array([3, 7])
    >>> zt[1, 0]
    array([2, 6])
    >>> zt[1, 1]
    array([4, 8])
    """
    x = np.array([
        [1, 2],
        [3, 4]
    ])
    y = np.array([
        [5, 6],
        [7, 8]
    ])
    z = np.dstack([x, y])
    return z, z.transpose(1, 0, 2)
