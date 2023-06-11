import cv2 as cv
import numpy as np
from torchvision import transforms
from hwr.data_proc.char_proc.utils import to_cv, cv_to_tensor
from hwr.utils.misc import named
from hwr.segmenter.lines import transforms as lin_seg_tx
import torchvision.transforms.v2 as t2


class NamedLambda:
    def __init__(self, f, name=None):
        self.txfn = transforms.Lambda(f)

        self.name = name
        if self.name is None and hasattr(f, "__name__"):
            self.name = f.__name__

        named(self, self.name or "anon")

    def __repr__(self):
        return f"NamedLambda({self.name})"

    def __call__(self, *args, **kwargs):
        return self.txfn(*args, **kwargs)


class OpenCvLambda(NamedLambda):
    def __init__(self, f, name=None):
        super().__init__(lambda x: cv_to_tensor(f(to_cv(x))), name=name)

    def __repr__(self):
        return f"OpenCvLambda({self.name})"


class TorchLambda(NamedLambda):
    def __init__(self, f, name=None):
        super().__init__(lambda x: to_cv(f(cv_to_tensor(x))), name=name)

    def __repr__(self):
        return f"OpenCvLambda({self.name})"


tx_squeeze = NamedLambda(lambda t: t.squeeze(), "Squeeze")
tx_negative = NamedLambda(lambda t: 1 - t, "Neg1")

# The first type of transform we need is one that loads the images to grayscale
# torch tensors we also remove the color channel and
# invert the black/white pixels
tx_base = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
        tx_squeeze,
        tx_negative,
    ]
)


# --- First we define our functions as numpy -> numpy
def binarize_otsu(x: np.array) -> np.array:
    _, xout = cv.threshold(x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return xout


def binarize_triangle(x: np.array) -> np.array:
    _, xout = cv.threshold(x, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)
    return xout


def erosion(x: np.array) -> np.array:
    return cv.erode(x, np.ones((3, 3)))


def cut_via_contours(image: np.array) -> np.array:
    # Bounding box extraction for largest contour (by area)
    contours, _ = cv.findContours(
        image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(contour)

    # Crop the image based on the bounding box
    character = image[y : y + h, x : x + w]

    return character


def cat_contours(image: np.array) -> np.array:
    # Bounding box extraction for largest contour (by area)
    to_cont = binarize_triangle(image)

    contours, _ = cv.findContours(
        to_cont, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(contour)

    # Crop the image based on the bounding box
    character = image[y : y + h, x : x + w]

    return character


cutter = lin_seg_tx.ProjectionCutter(
    complement=0,
    pad=0,
    margin=0,
    min_fraction=0.1,
)


def square_pad(x: np.array) -> np.array:
    h, w = x.shape
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    return cv.copyMakeBorder(
        x, vp, vp, hp, hp, cv.BORDER_CONSTANT, value=[0, 0, 0]
    )


# --- Then we can wrap them to become torch.Tensor -> torch.Tensor
tx_outsu_bin = OpenCvLambda(binarize_otsu, name="OtsuBin")
tx_triang_bin = OpenCvLambda(binarize_triangle, name="TirangBin")
tx_cont_cut = OpenCvLambda(cut_via_contours, name="Cut_0")
tx_cont = OpenCvLambda(cat_contours, name="Cut_1")
tx_square_pad = OpenCvLambda(square_pad, name="square_pad")
tx_proj_cut = OpenCvLambda(cutter, name="proj_cut")
tx_prep_erode = OpenCvLambda(erosion, name="eroded")

tx_unsqueeze = NamedLambda(lambda t: t.unsqueeze(0), "Unsqueeze")

tx_prep_otsu = named(
    transforms.Compose([tx_outsu_bin, tx_cont_cut, tx_square_pad]),
    "Otsu+cont+square",
)
tx_prep_tri = named(
    transforms.Compose([tx_triang_bin, tx_cont_cut, tx_square_pad]),
    "Triang+cont+square",
)


tx_prep_cut = named(transforms.Compose([tx_cont, tx_square_pad]), "Only_cut")


tx_prep_erode = named(
    transforms.Compose([tx_prep_cut, tx_prep_erode]),
    "erode",
)

tx_prep_zoom = named(
    transforms.Compose(
        [
            tx_prep_cut,
            tx_unsqueeze,
            t2.RandomZoomOut(p=1, side_range=(1, 2)),
            tx_squeeze,
        ]
    ),
    "zoom",
)

tx_prep_rotation = named(
    transforms.Compose(
        [
            tx_prep_cut,
            tx_unsqueeze,
            transforms.RandomRotation(22.5),
            tx_squeeze,
        ]
    ),
    "rotated",
)

tx_prep_perspective = named(
    transforms.Compose(
        [
            tx_prep_cut,
            tx_unsqueeze,
            transforms.RandomPerspective(0.5, 1),
            tx_squeeze,
        ]
    ),
    "perspective",
)


tx_noop = NamedLambda(lambda t: t, "Noop")

char_transforms = [
    transforms.Compose([tx_base, tx])
    for tx in [
        tx_noop,
        tx_cont,
        tx_prep_erode,
        tx_prep_rotation,
        tx_prep_perspective,
        tx_prep_zoom,
    ]
]


train_txs = transforms.Compose(
    [
        transforms.Resize(
            (28, 28),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)

real_txs = TorchLambda(
    transforms.Compose(
        [
            tx_prep_cut,
            tx_unsqueeze,
            transforms.Resize(
                (28, 28),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            tx_squeeze,
        ]
    )
)
