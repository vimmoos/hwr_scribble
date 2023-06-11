from dataclasses import dataclass, field
from functools import partial
from logging import getLogger
from typing import List, Optional, Tuple, Union
import cv2
import deskew
import matplotlib.pyplot as plt
import numpy as np
import peakdetect
from deskew import determine_skew
from numpy.typing import NDArray
from scipy.signal import savgol_filter
from skimage.filters import threshold_otsu
from skimage.transform import rotate
from skimage import morphology
from benedict import OrderedBeneDict

from .types import ImgTx

log = getLogger(__name__)


# ===================== Preprocessing and misc transforms =========================
@dataclass
class ComposeTx:
    """Sequential composition of transforms NDArray -> NDArray"""

    steps: List[ImgTx]

    def __call__(self, img: NDArray) -> NDArray:
        for step in self.steps:
            img = step(img)
        return img


@dataclass
class SimpleResizer:
    """Resize image to a some fraction of the original size"""

    percent_resize: float = 0.5
    cv2_interp: int = cv2.INTER_CUBIC

    def __call__(self, img: NDArray) -> NDArray:
        # print('Original Dimensions : ', img.shape)
        width = int(img.shape[1] * self.percent_resize)
        height = int(img.shape[0] * self.percent_resize)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized


def projections(X, axis=1, max_int_value=255):
    """Computes the projection profile along specified axis.
    Values along the axis are summed up and normalized by the maximum possible value,
    that is an entry of the returned profile is 1 if all pixels had `max_int_value` intensity
    along that row (or column)

    axis=1 will compute HPP
    axis=0 will compute VPP
    """
    return np.sum(X, axis=axis) / (max_int_value * X.shape[axis])


# def show_proj(X, axis=1):
#     plt.plot(projections(X, axis=axis))


def get_binary(img, th_fn=threshold_otsu, white=255):
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    thresh = th_fn(img)
    binary = (img > thresh) * 1
    return np.array(white * binary, dtype=np.uint8)


def proj_cut(
        X,
        complement=255,
        min_fraction=0.001,
        margin: int = 5,
        max_int_value: int = 255,
        pad: Optional[int] = None,
        pad_value: int = 255,
):
    """Cut the given image based on projection profile heuristic.
    an area is considered empty/whitespace if it has less than `min_fraction` of the max intensity points.
    """
    ret = X.copy()
    X = get_binary(complement - X if complement else X, white=max_int_value)

    h_proj = projections(X, axis=1, max_int_value=max_int_value)
    (h_nonzeros,) = np.where(h_proj > min_fraction)

    v_proj = projections(X, axis=0, max_int_value=max_int_value)
    (v_nonzeros,) = np.where(v_proj > min_fraction)

    if len(h_nonzeros) > 1:
        h_min, h_max = h_nonzeros[0], h_nonzeros[-1]
    else:
        h_min, h_max = 0, X.shape[0]

    if len(v_nonzeros) > 1:
        v_min, v_max = v_nonzeros[0], v_nonzeros[-1]
    else:
        v_min, v_max = 0, X.shape[1]

    # print(h_min, h_max, v_min, v_max)

    cut = ret[
          max(h_min - margin, 0): h_max + margin,
          max(v_min - margin, 0): v_max + margin,
          ]

    if pad is not None:
        return small_padding(cut, p=pad, value=pad_value)
    return cut


@dataclass
class ProjectionCutter:
    """Cut image main region based on projections"""

    complement: int = 255
    min_fraction: float = 0.001
    margin: int = 5
    max_int_value: int = 255
    pad: Optional[int] = None
    pad_value: int = 255

    def __call__(self, img: NDArray) -> NDArray:
        return proj_cut(
            img,
            complement=self.complement,
            min_fraction=self.min_fraction,
            margin=self.margin,
            max_int_value=self.max_int_value,
            pad=self.pad,
            pad_value=self.pad_value,
        )


# Doing this trivially is not good enought for our documents:
# def deskew_image(img, complement=255):
#     """Deskew the given image"""
#     _X = complement - img
#     angle = determine_skew(_X)
#     return complement - (complement * rotate(_X, angle))

# Most of the error come from the fact that the number default # of peak estimates
# is wrong. We therefore implement a custom logic to estimate the number of peaks.
# this involves taking an image and study its projection profile and count the peaks
# to estimate a number of lines to fit to find the correction angle


def _custom_deskew(
        X: NDArray,
        prep_tx: Optional[ImgTx] = None,
        smoot_hpp_win_len=20,
        smoot_polyorder=1,
        peak_lookahead=5,
        peak_delta=0.05,
        max_angle=10,
        min_skew_accepted=0.001,
):
    """Backbone functional definition"""
    _X = X.copy()
    _X = prep_tx(_X) if prep_tx is not None else _X
    hpp = projections(_X, axis=1)
    s_hpp = savgol_filter(
        hpp, window_length=smoot_hpp_win_len, polyorder=smoot_polyorder
    )
    max_peaks, _ = peakdetect.peakdetect(
        s_hpp, lookahead=peak_lookahead, delta=peak_delta
    )
    skew = determine_skew(
        X, num_peaks=len(max_peaks), max_angle=max_angle, min_angle=-max_angle
    )
    if skew is None or 0 <= skew < min_skew_accepted:
        log.info("No skew determined")
        return X, skew
    log.info("Skew detected: %s", skew)
    return 255 * (rotate(X, skew)), skew


def small_padding(img, p=5, value=255):
    """Add equal padding on all sides of the given image"""
    return cv2.copyMakeBorder(
        img, p, p, p, p, cv2.BORDER_CONSTANT, value=value
    )


def tx(txfn: Union[ImgTx, None], X: NDArray):
    if txfn is None:
        return X
    return txfn(X)


def min_max_scale(xs):
    _min = xs.min()
    _max = xs.max()
    support = _max - _min
    return (xs - _min) / support


@dataclass
class _DeskewAndPadTx:
    """Apply a deskewing rotation to the image plus a small padding.

    This is the generic object wrapper implementation

    Deskew ensures lines are optimally horizontal, padding ensured the pathfinding can find a route
    for characters that touch the borders.

    """

    # general
    white: int = 255
    pad: int = 5
    prep_tx: Optional[ImgTx] = None

    # peak detect
    smoot_hpp_win_len: int = 20
    smoot_polyorder: int = 1
    peak_lookahead: int = 5
    peak_delta: float = 0.05

    # deskew
    max_angle: int = 10
    deskew_min_dev_degrees: float = 1.0
    deskew_sigma: float = 3.0
    min_skew_accepted: float = 0.001

    input_white_on_black: bool = False

    speedup_downscaler: Optional[ImgTx] = None

    s: OrderedBeneDict = field(default_factory=OrderedBeneDict, init=False)

    def __call__(self, img: NDArray) -> NDArray:
        self.s.input_dtype = img.dtype
        self.s.input = img.copy()

        # invert the image if the input is not white on black
        self.s.X = img if self.input_white_on_black else ~img
        self.s.downscaled = tx(self.speedup_downscaler, self.s.X)
        self.s.prepped = tx(self.prep_tx, self.s.downscaled)
        self.s.hpp = np.sum(
            self.s.prepped, axis=1
        )  # projections(self.s.prepped, axis=1)
        self.s.s_hpp = min_max_scale(
            savgol_filter(
                self.s.hpp,
                window_length=self.smoot_hpp_win_len,
                polyorder=self.smoot_polyorder,
            )
        )

        max_peaks, min_peaks = peakdetect.peakdetect(
            self.s.s_hpp, lookahead=self.peak_lookahead, delta=self.peak_delta
        )

        self.s.max_peaks, self.s.min_peak = np.array(max_peaks), np.array(
            min_peaks
        )

        self.s.skew = determine_skew(
            self.s.prepped,
            sigma=self.deskew_sigma,
            min_deviation=self.deskew_min_dev_degrees,
            num_peaks=len(self.s.max_peaks),
            max_angle=self.max_angle,
            min_angle=-self.max_angle if self.max_angle is not None else None,
        )

        if self.s.skew is None or 0 <= self.s.skew < self.min_skew_accepted:
            log.info("No skew determined")
            self.s.rotated = self.s.X
        else:
            log.info("Skew detected: %s", self.s.skew)
            self.s.rotated = rotate(
                self.s.X,
                angle=self.s.skew,
                preserve_range=True,
                cval=0,
                resize=True,
            ).astype(self.s.input_dtype)

        self.s.padded = small_padding(self.s.rotated, p=self.pad, value=0)

        self.s.output = (
            self.s.padded if self.input_white_on_black else ~self.s.padded
        )

        return self.s.output.copy()

    def diagnose(self):
        return deskew.determine_skew_debug_images(
            self.s.prepped,
            sigma=self.deskew_sigma,
            min_deviation=self.deskew_min_dev_degrees,
            num_peaks=len(self.s.max_peaks),
            max_angle=self.max_angle,
            min_angle=-self.max_angle if self.max_angle is not None else None,
        )

    def diagnose_peaks(self):
        fig, axs = plt.subplots(nrows=2)
        axs[0].plot(self.s.s_hpp)
        axs[0].scatter(
            self.s.max_peaks[:, 0],
            self.max_peaks[:, 1],
            marker="x",
            color="red",
        )
        axs[1].imshow(self.prepped, cmap="Greys_r")
        for x in self.s.max_peaks:
            axs[1].axvline(x=x[0])


@dataclass
class DeskewAndPadTx(_DeskewAndPadTx):
    """Apply a deskewing rotation to the image plus a small padding.

    This uses dilation preprocssing

    Deskew ensures lines are optimally horizontal, padding ensured the pathfinding can find a route
    for characters that touch the borders.

    """

    dil_kernel_size: Union[Tuple[int, int], NDArray] = (1, 20)
    dil_its: int = 1

    def __post_init__(self):
        if isinstance(self.dil_kernel_size, tuple):
            dil_kernel = np.ones(shape=self.dil_kernel_size)
        elif isinstance(self.dil_kernel_size, np.ndarray):
            dil_kernel = self.dil_kernel_size
        else:
            raise Exception(
                "The dil_kernel_size param must be tuple (to generate ones) or an explicity NDArray kernel"
            )

        self.prep_tx = lambda X: cv2.dilate(
            X, kernel=dil_kernel, iterations=self.dil_its
        )


# Alternative deskewing


@dataclass
class DeskewAndPadTx2(_DeskewAndPadTx):
    """Apply a deskewing rotation to the image plus a small padding.

    This uses smooth-bin-medial-axis approach.

    Deskew ensures lines are optimally horizontal, padding ensured the pathfinding can find a route
    for characters that touch the borders.

    """

    smooth_k_size: Tuple[int, int] = (100, 20)
    peak_delta: float = 0.3
    max_angle: float = 10
    deskew_min_dev_degrees: float = 1.0
    deskew_sigma: float = 1.0
    bin_fn: ImgTx = get_binary

    def __post_init__(self):
        self.prep_tx = ComposeTx(
            [
                partial(cv2.blur, ksize=self.smooth_k_size),
                self.bin_fn,
                partial(morphology.medial_axis),
            ]
        )


# Alternative deskewing


# --- Transforms on paths ---
def resize_paths(paths: List[NDArray], factor) -> List[NDArray]:
    ret = [np.int32(path * factor) for path in paths]
    # raise Exception("BREAK", ret)
    return ret
