import numpy as np
from itertools import tee
from typing import Iterable
import torch as th
from numpy.typing import NDArray
from hwr.types import Div0RiskError


def sel_keys(d, ks):
    return {k: v for k, v in d.items() if k in ks}


def shift_in_unit(xs):
    _min = xs.min()
    _max = xs.max()
    support = _max - _min
    if support == 0:
        raise Div0RiskError("Trying to shift in unit uniform data")
    return (xs - _min) / support


def get_all_labels(dataset):
    return np.array([x[1] for x in dataset])


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_random_indexs(dataset: Iterable, n: int):
    return np.random.choice(range(len(dataset)), size=n)


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


def named(obj, name):
    setattr(obj, "__name__", name)
    return obj


def joinit(iterable, delimiter):
    """Intersperese iterable with delimiter"""
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x
