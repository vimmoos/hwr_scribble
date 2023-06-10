from matplotlib import patches
import distinctipy


def draw_image_with_slices(img, slices, ax, classes=None, cmap="Greys_r"):
    ax.imshow(img, cmap=cmap)
    _colors = distinctipy.get_colors(len(slices))
    for i, sl in enumerate(slices):
        ax.add_patch(
            patches.Rectangle(
                (sl.start, 0),
                width=sl.stop - sl.start,
                height=img.shape[0],
                alpha=0.5,
                facecolor=_colors[i],
            )
        )
        if classes is not None:
            ax.text(sl.start, -1, classes[i])
