import numpy as np
import cv2
from segmenter.lines import transforms as lin_seg_tx


def square_pad(X, pad_value=0):
    height, width = X.shape
    max_wh = np.max([width, height])
    wp = int((max_wh - width) / 2)
    hp = int((max_wh - height) / 2)
    padding = [hp, hp, wp, wp]
    return cv2.copyMakeBorder(X, *padding, cv2.BORDER_CONSTANT, pad_value)


def clf_tx(X, w_h=28):
    X = lin_seg_tx.ProjectionCutter(complement=0, pad=0, margin=0)(X)
    X = square_pad(X)
    return cv2.resize(X, dsize=(w_h, w_h), interpolation=cv2.INTER_NEAREST)
