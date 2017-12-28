#!/usr/bin/env python

"""
performance.py

Use these methods to measure and compare the performance of different approaches.
"""

import time

def timed(task):
    """A decorator used for timing function calls."""
    def _decorated(fn):
        def _fn(*args, **kwargs):
            print('[o] %s has begun.' % task)
            start_time = time.time()
            result = fn(*args, **kwargs)
            print('[o] %s has ended (%.4fs elapsed).' % (task, time.time() - start_time))
            return result
        return _fn
    return _decorated
