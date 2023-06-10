import numpy as np
from itertools import tee
from typing import Iterable
import torch as th
from numpy.typing import NDArray


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


def batcher(
    iterable: Iterable,
    batch_size: int = 64,
    appender: callable = lambda acc, y: [y] if acc is None else acc + [y],
):
    batch = None
    for el in iterable:
        batch = appender(batch, el)
        if len(batch) == batch_size:
            yield batch
            batch = None
    if batch is not None:
        yield batch


def to_tensor(img: NDArray, max_value=255):
    """Given an image (H,W) returns a normalized (between [0,1])tensor (1,H,W)"""
    return th.Tensor(img / max_value).unsqueeze(0)
