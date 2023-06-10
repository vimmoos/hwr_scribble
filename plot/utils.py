import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.gridspec import GridSpec


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


def joinit(iterable, delimiter):
    """Intersperese iterable with delimiter"""
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x


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
