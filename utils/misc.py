import numpy as np
from itertools import tee


def sel_keys(d, ks):
    return {k: v for k, v in d.items() if k in ks}


def shift_in_unit(xs):
    _min = xs.min()
    _max = xs.max()
    support = _max - _min
    return (xs - _min) / support


def get_all_labels(dataset):
    return np.array([x[1] for x in dataset])


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
