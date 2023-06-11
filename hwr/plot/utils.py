from typing import Union, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from hwr.utils.misc import joinit


def show_bw(array, figsize=(10, 10), cmap="Greys_r", color_bar=False):
    fig, ax = plt.subplots(figsize=figsize)
    r = ax.imshow(array, cmap=cmap)
    if color_bar:
        fig.colorbar(r)


def line_splits_view(image, line_images, cmap="Greys_r", figsize=(8, 8)):
    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(nrows=len(line_images), ncols=2, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(image, cmap=cmap)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for i, l in enumerate(line_images):
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(l, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])

    return fig


def imshow_rows(ims, cmap="Greys_r", figsize=(8, 8)):
    fig = plt.figure(figsize=figsize)
    for i, l in enumerate(ims):
        ax = fig.add_subplot(len(ims), 1, i + 1)
        ax.imshow(l, cmap=cmap)
    fig.tight_layout()
    return fig


def resize_height(img, new_height: int):
    """Resize to new height maintaing aspect ratio"""
    width = img.shape[1]
    height = img.shape[0]
    whr = width / max(height, 1)
    new_width = max(int(whr * new_height), 1)
    return cv2.resize(
        img, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC
    )


def make_hstack_image(word_imgs, resize_h=50, sep_width=10):
    """Stich together the given `word_imgs` to display a coherent size line with spaces as blue strips"""
    res_words = [
        cv2.cvtColor(
            resize_height(img.copy(), new_height=resize_h), cv2.COLOR_GRAY2BGR
        )
        for img in word_imgs
    ]
    sep_img = np.zeros((resize_h, sep_width, 3), np.uint8)
    sep_img[:] = (0, 0, 255)
    with_seps = list(joinit(res_words, sep_img))
    vis = np.concatenate(with_seps, axis=1)
    return vis


def show_lines(lines):
    fig, axs = plt.subplots(nrows=len(lines), figsize=(10, 10))
    for i, line in enumerate(lines):
        axs[i].imshow(line)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    return fig


def draw_contours_bw(img, contours, cont_color=(0, 255, 0)):
    lin_show = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(lin_show, contours, -1, cont_color, thickness=3)
    return lin_show


def resize_width(img, new_width: int):
    """Resize to new height maintaing aspect ratio"""
    width = img.shape[1]
    height = img.shape[0]
    hwr = height / width
    new_height = max(int(hwr * new_width), 1)
    return cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)


def make_word_display_line(line_imgs, resize_w=500, sep_height=10):
    res_lines = [cv2.cvtColor(resize_width(img.copy(), new_width=resize_w), cv2.COLOR_GRAY2BGR) for img in line_imgs]
    sep_img = np.zeros((sep_height, resize_w, 3), np.uint8)
    sep_img[:] = (0, 0, 255)
    with_seps = list(joinit(res_lines, sep_img))
    vis = np.concatenate(with_seps, axis=0)
    return vis


def pad_to_uniform_strips(strips,
                          axis: Union[Literal["height", "width"], int] = "height",
                          pad_value=0):
    if isinstance(axis, str):
        axis = 0 if axis == "height" else 1

    max_s = max([s.shape[axis] for s in strips])
    padded_strips = []
    for strip in strips:
        if strip.shape[axis] == max_s:
            padded_strips.append(strip)
            continue
        # then we have to pad
        s_diff = max_s - strip.shape[axis]
        # pad top if height, left if width
        pad_args = (s_diff, 0, 0, 0) if axis == 0 else (0, 0, s_diff, 0)

        padded_strips.append(cv2.copyMakeBorder(strip, *pad_args,
                                                cv2.BORDER_CONSTANT,
                                                pad_value))
    return padded_strips
