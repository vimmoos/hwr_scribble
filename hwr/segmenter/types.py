from typing import (
    Callable,
    TypedDict,
    Tuple,
    Dict,
    List
)
from numpy.typing import NDArray
from skimage import measure
from collections import OrderedDict


class ComponentDescriptor(TypedDict):
    label: int

    pix_count: int
    pix_perc: float

    h_bounds: Tuple[int, int]
    w_bounds: Tuple[int, int]
    area: int
    xy_center: Tuple[int, int]

    comp_map: NDArray
    cut: NDArray
    strip_cut: NDArray
    poly: NDArray

    props: measure._regionprops.RegionProperties


class ComponentsMeta(TypedDict):
    X: NDArray
    blob_map: NDArray
    blob_lables: NDArray


Cv2Contours = NDArray
Cv2ContoursTx = Callable[[Cv2Contours], Cv2Contours]

ContoursFinder = Callable[[NDArray], List[Cv2Contours]]
ComponentsSorter = Callable[[Dict[int, ComponentDescriptor]], List[ComponentDescriptor]]

LineRecons = Callable[[NDArray,
                       OrderedDict[int, NDArray],
                       Dict[int, ComponentDescriptor],
                       OrderedDict[int, List[int]]],
OrderedDict[int, NDArray]]

WordSegmentator = Callable[[NDArray], List[NDArray]]
