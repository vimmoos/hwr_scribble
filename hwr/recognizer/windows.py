import numpy as np
from typing import Optional, Callable, Union, Tuple
from numpy.typing import NDArray
from dataclasses import dataclass
from functools import partial


class WindowOutOfBounds(Exception):
    def __init__(self, *args):
        super().__init__("Window is out of bounds", *args)


SliceAcceptor = Callable[[slice, int, int, NDArray], Union[slice, None]]


def accept_all_slices(sl, pos, w, X):
    return sl


def accept_slices_min_width(sl, pos, w, X, *, min_width):
    if sl.stop - sl.start < min_width:
        return None
    return sl


def accept_slices_bounce(sl, pos, w, X):
    return slice(0, w)


@dataclass
class WindowSlider:
    width: int
    img: NDArray
    pos: int = 0

    img_tx: Optional[Callable[[NDArray], NDArray]] = None

    last_slice_accept: SliceAcceptor = accept_all_slices

    def __call__(self):
        if self.pos == -np.inf:
            raise WindowOutOfBounds()

        new_pos = self.img.shape[1] - self.pos

        sl = slice(new_pos - self.width, new_pos)

        if (self.pos + self.width) > self.img.shape[1]:
            last_slice = slice(0, new_pos)
            sl = self.last_slice_accept(
                last_slice,
                self.pos,
                self.width,
                self.img,
            )
            self.pos = -np.inf
            if sl is None:
                raise WindowOutOfBounds(last_slice)

        self.ret = self.img[:, sl]

        return self.ret if self.img_tx is None else self.img_tx(self.ret)

    def __lshift__(self, offset):
        self.pos += offset

    def __rshift__(self, offset):
        self.pos -= offset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val, WindowOutOfBounds):
            return True

    def freeze(self) -> Tuple[int, int, NDArray]:
        return (self.width, self.pos, self.ret.copy())

    def load(self, width: Optional[int] = None, pos: Optional[int] = None):
        self.width = width or self.width
        self.pos = pos or self.pos
        return self


# BECAREFULL !!!!!!!!!!!!!!!!!!!!!!! CLF_TX


# def simple_fixed_slider(X, w=56):
#     return WindowSlider(width=w, img=X, img_tx=clf_tx)
#
#
# def simple_fixed_slider2(X, w=56):
#     return WindowSlider(
#         width=w,
#         img=X,
#         img_tx=clf_tx,
#         last_slice_accept=partial(accept_slices_min_width, min_width=10),
#     )
