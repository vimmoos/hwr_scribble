import numpy as np
from hwr.utils.misc import shift_in_unit
import cv2
from typing import List
from numpy.typing import NDArray


def bound_to_regions(bounds):
    return [(x[1], y[0]) for x, y in zip(bounds, bounds[1:])]


def splits_from_hpp(hpp, peak_group_filter=None):
    hpp_norm = shift_in_unit(hpp)
    zeros, = np.where(hpp_norm < 0.05)
    fo = np.diff(zeros)
    edges = np.where(fo > 1)[0].flatten() + 1
    groups = np.split(zeros, edges)

    groups_f = groups if peak_group_filter is None else peak_group_filter(groups)
    # print(groups_f)

    bounds = []
    if 0 not in groups_f[0]:
        bounds.append((0, 0))
    for group in groups_f:
        bounds.append((group[0], group[-1]))
    if len(hpp) - 1 not in groups_f[-1]:
        bounds.append((len(hpp) - 1, len(hpp) - 1))

    return bound_to_regions(bounds)


def simple_split_words(line):
    blurred_line = cv2.GaussianBlur(line, ksize=(51, 51), sigmaX=4, sigmaY=10)
    hpp = np.sum(blurred_line, axis=0)
    return splits_from_hpp(hpp)


def get_word_cuts(img, cuts):
    word_imgs = []
    for cut in cuts:
        word_imgs.append(img[:, cut[0]:cut[1] + 1])
    return word_imgs


def segment_words(img: NDArray, right_to_left=True) -> List[NDArray]:
    cuts = simple_split_words(img)
    words = get_word_cuts(img, cuts)
    return list(reversed(words)) if right_to_left else words
