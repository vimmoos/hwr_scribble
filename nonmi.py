from collections import Counter
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from itertools import groupby
from itertools import islice
from pathlib import Path
from pprint import pprint
from typing import Any
from typing import Iterable, Generator
from typing import Optional, Callable, Tuple, Union, List

import cv2
import distinctipy
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import wandb
from benedict import BeneDict
from benedict import OrderedBeneDict
from hwr2.common import dss_charset
from hwr2.common import vizutils
from hwr2.tasks.segment_lines import transforms as lin_seg_tx
from hwr2.tasks.uq import uq_utils
from hwr2.tasks.uq.autoencoder import Autoencoder
from hwr2.tasks.uq.utils import shift_in_unit
from matplotlib import patches
from numpy.typing import NDArray
from peakdetect import peakdetect


def sel_keys(d, ks):
    return {k:v for k,v in d.items() if k in ks}

run = wandb.init()
artifact = run.use_artifact('drl42/HWR_UQ/model-3387c7fi:v0', type='model')
artifact_dir = artifact.download()

import os
os.listdir(artifact_dir)

checkpoint = Path(artifact_dir) / "model.ckpt"
model = Autoencoder.load_from_checkpoint(checkpoint)

model.cpu()


@dataclass
class Word:
    pagename: str
    pageno: int
    lineno: int
    wordno: int
    img: NDArray = field(repr=False)
    labels: List[int] = field(default_factory=list)
    meta: BeneDict = field(default_factory=BeneDict)

    def sort_key(self):
        return (self.pageno, self.lineno, self.wordno)

    def decode(self, decoder):
        return [decoder(y) for y in self.labels]

@dataclass
class WordLoader:
    root: Path

    page_glob: str = "page-*.d"

    line_dir: str = "lines.d"
    line_glob: str = "line-*.d"

    word_dir: str = "words.d"
    word_glob: str = "word-*.png"

    load_tx: Optional[Callable[[NDArray], NDArray]] = None

    imread_mode: int = cv2.IMREAD_GRAYSCALE

    def load_img(self, path):
        img = cv2.imread(str(path), self.imread_mode)
        return self.load_tx(img) if self.load_tx is not None else img

    def list_pages(self):
        return list(sorted(self.root.glob(self.page_glob)))

    def list_lines(self, pageno):
        page_dir = self.list_pages()[pageno]
        return list(sorted((page_dir / self.line_dir).glob(self.line_glob)))

    def list_word_paths(self, pageno, lino):
        line_dir = self.list_lines(pageno)[lino]
        word_paths = sorted((line_dir / self.word_dir).glob(self.word_glob))
        return word_paths

    def list_words(self, pageno, lino):
        word_paths = self.list_word_paths(pageno, lino)
        for word_path in word_paths:
            yield self.load_img(word_path)

    def __iter__(self):
        for pageno, page_path in enumerate(self.list_pages()):
            for lino, line_path in enumerate(self.list_lines(pageno)):
                for wordno, img in enumerate(self.list_words(pageno, lino)):
                    yield Word(pagename=page_path.name,
                               pageno=pageno,
                               lineno=lino,
                               wordno=wordno,
                               img=img)

    def get_one_word(self, pageno, lino, wordno):
        return self.load_img(self.list_word_paths(pageno, lino)[wordno])

def draw_image_with_slices(img, slices, ax, classes=None, cmap="Greys_r"):
    ax.imshow(img, cmap=cmap)
    _colors = distinctipy.get_colors(len(slices))
    for i, sl in enumerate(slices):
        ax.add_patch(patches.Rectangle((sl.start, 0), width=sl.stop-sl.start, height=img.shape[0] , alpha=0.5, facecolor=_colors[i] ))
        if classes is not None:
            ax.text(sl.start, -1, classes[i])


class WindowOutOfBounds(Exception):
    def __init__(self, *args):
        super().__init__("Window is out of bounds", *args)


def accept_all_slices(sl, pos, w, X):
    return sl


def accept_slices_min_width(sl, pos, w, X, *, min_width):
    if sl.stop - sl.start < min_width:
        return None
    return sl


def accept_slices_bounce(sl, pos, w, X):
    return slice(0, w)


SliceAcceptor = Callable[[slice, int, int, NDArray], Union[slice, None]]

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
            sl = self.last_slice_accept(last_slice, self.pos, self.width, self.img)
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

def simple_fixed_slider(X, w=56):
    return WindowSlider(width=w, img=X, img_tx=clf_tx)

def simple_fixed_slider2(X, w=56):
    return WindowSlider(width=w, img=X, img_tx=clf_tx, last_slice_accept=partial(accept_slices_min_width, min_width=10))

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


class ErrorUnsgementableWord(Exception):
    pass


@dataclass
class WindowOverSegment:
    pixel_lookahed: int = 3

    s: OrderedBeneDict = field(default_factory=OrderedBeneDict)

    WORD_DISCARD_BIN: Optional[List["WindowOverSegment"]] = None

    def __call__(self, X):
        try:
            self.s.input = X
            self.s.vpp = np.sum(X, axis=0)
            self.s.vpp_unit = shift_in_unit(self.s.vpp)
            _, min_peaks = peakdetect(self.s.vpp_unit, lookahead=self.pixel_lookahed)
            self.s.min_peaks_xy = np.array(min_peaks)
            self.s.min_peaks_x = self.s.min_peaks_xy[:, 0]

            self.s.bounds = [0] + list(self.s.min_peaks_x) + [X.shape[1]]
            self.s.x_slices = [slice(int(mmin), int(mmax)) for mmin, mmax
                               in list(reversed(list(zip(self.s.bounds, self.s.bounds[1:]))))]

            self.s.img_slices = [X[:, sl] for sl in self.s.x_slices]

        except Exception as e:
            # raise ErrorUnsgementableWord(e, self)
            self.WORD_DISCARD_BIN.append(self)
            self.s.img_slices = []
            self.s.x_slices = []

        return self.s.img_slices.copy()

    def get_current_slices(self):
        return self.s.x_slices.copy()

    def show_debug(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        tit = "Window oversegment debug"
        ax1.plot(self.s.input.shape[0] * (1 - self.s.vpp_unit))

        ax1.imshow(self.s.input)
        if len(self.s.min_peaks_xy):
            ax1.scatter(self.s.min_peaks_xy[:, 0], self.s.input.shape[0] * (1 - self.s.min_peaks_xy[:, 1]), marker="x",
                        color="red")
            ax2.imshow(vizutils.make_hstack_image(reversed(self.s.img_slices), sep_width=1))
        else:
            tit += " (no min-peaks)"
            ax2.set_axis_off()
        fig.suptitle(tit)
        return fig

# def mk_slice_seq(slices, max_slices, slice_tx):
#     return [slice_tx(np.hstack(list(reversed(slices[0:n])))) for n in range(1,max_slices+1)]

def img_batch_to_tensor_batch(Xs):
    return th.Tensor(np.array(Xs)).unsqueeze(1)



def slices_to_bound(slices):
    start = min([s.start for s in slices])
    end = max([s.stop for s in slices])
    return start, end

def merge_slices(slices):
    start, end = slices_to_bound(slices)
    return slice(start,end)


def get_chars2(X,
               model,
               window_tx: Callable[[NDArray], NDArray],
               MIN_WIN=10,
               MAX_WIN=40,
               ):
    wos = WindowOverSegment()
    # pass the image to to obtain a list of images
    all_slices = list(wos(X))
    all_slices_meta = wos.get_current_slices()

    print(f"Starting with {len(all_slices)} slices")
    # now we must pass frames to classifer untill we exaust them
    # each time we pass the first N windows to

    selections = []  # this is the returned object

    # create coupled dequeues
    img_slices = deque(all_slices.copy())
    img_slices_meta = deque(all_slices_meta.copy())

    it = -1
    # meta_slices = all_slices_meta.copy()
    while img_slices:
        it += 1
        print(f"------- Iteration {it} -------")
        batch = []
        batch_meta = []

        # take the first in queue, this is always consumes from the queue so this eventually terminates
        curr = img_slices.popleft()
        curr_meta = img_slices_meta.popleft()

        parts = deque([])  # accumulate the other used parts
        # the first one will be the last to appear, so they are in the
        # order in which they should be put back in (via append left)
        # in the main queue (img_slices) if they end up not being used
        # which means we can 'append-left' back if not used in this order
        parts_meta = deque([])

        # untill we reach above MIN_WIN then keep stacking
        while curr.shape[1] < MIN_WIN and img_slices:
            print(f"Window w={curr.shape[1]} below MIN_WIN={MIN_WIN}, enlarging...")

            _next = img_slices.popleft()
            _next_meta = img_slices_meta.popleft()

            parts.appendleft(_next)
            parts_meta.appendleft(_next_meta)

            curr = np.hstack((_next, curr))
            curr_meta = merge_slices((_next_meta, curr_meta))

        # if the img_slices finished before reaching MIN_WIN
        # then curr contains what was there, but we don't produce the frame at all
        if curr.shape[1] < MIN_WIN:
            print("Last window below MIN_WIN")
            break

        print("Created first batch frame, consumed slices:", len(parts))
        batch.append(curr)
        batch_meta.append(curr_meta)

        # note that the first frame is allowed to escape MAX_WIN, but will not produce attionals

        print("Generating additional fram up to max width")

        # while we are belew max and can take other slices
        while curr.shape[1] <= MAX_WIN and img_slices:
            _next = img_slices.popleft()  # take the next slice
            _next_meta = img_slices_meta.popleft()

            new_curr = np.hstack((_next, curr))
            new_curr_meta = merge_slices((_next_meta, curr_meta))

            # if with this we go out of bounds then put it back
            if new_curr.shape[1] > MAX_WIN:
                img_slices.appendleft(_next)  # put back
                img_slices_meta.appendleft(_next_meta)
                break  # we are done

            parts.appendleft(_next)  # register the part
            parts_meta.appendleft(_next_meta)

            print(f"Creating {len(batch)}-th batch frame")

            batch.append(new_curr)
            batch_meta.append(new_curr_meta)

            curr = new_curr
            curr_meta = new_curr_meta

        # ok now we have a batch with at least 1 item, possibly more,
        # where items have min width MIN_WIN and max MAX_WIN

        print(f"Preparing batch of size={len(batch)} valid slices")
        converted_frames = [window_tx(x) for x in batch]
        tensor_batch = img_batch_to_tensor_batch([frame / 255 for frame in converted_frames])
        print("Created tensor batch: ", tensor_batch.shape)

        print("Running classifier...")
        with uq_utils.confidences(model, n=100, only_confidence=False) as f:
            rvotes, _, imgs = f(tensor_batch)

        # extract the predicted classes
        cls = [Counter(x).most_common(1)[0][0] for x in rvotes]
        # extract the relative support for the selected classes
        rvalues = np.array([max(rvote.values()) for rvote in rvotes])

        # argmax on the confidence, this tells us what slice index to chose
        # 0 being use just the first frame, and e.g. 1 use the first 2 and so on
        chosen_slice_index = np.argmax(np.array(rvalues))
        # chosen_slice = slicez[chosen_slice_index]
        chosen_conf = rvalues[chosen_slice_index]
        chosen_label = cls[chosen_slice_index]
        chosen_bounds = batch_meta[chosen_slice_index]

        # img_bounds = slices_to_bound(meta_slices[slicez[chosen_slice_index]])

        print(f"Chosen slice: #{chosen_slice_index} conf = {chosen_conf} label = {chosen_label}")
        selections.append(BeneDict({"char": batch[chosen_slice_index],
                                    # "tensor": tensor_batch[chosen_slice_index],
                                    "frame": converted_frames[chosen_slice_index],
                                    "class": chosen_label,
                                    "conf": chosen_conf,
                                    "bounds": chosen_bounds,
                                    "recons": imgs[chosen_slice_index],
                                    # what was this chosen among?
                                    "meta": {
                                        "batch": batch,
                                        "frames": converted_frames,
                                        "clss": cls,
                                        "rvalues": rvalues,
                                        "avg_recons": np.mean(imgs, axis=0)
                                    }
                                    }))

        # put those that where not chosen
        _put_back_idx = -chosen_slice_index if chosen_slice_index else None
        put_back = list(parts)[:_put_back_idx]
        put_back_meta = list(parts_meta)[:_put_back_idx]

        for part, part_meta in zip(put_back, put_back_meta):
            img_slices.appendleft(part)
            img_slices_meta.appendleft(part_meta)

        print(f"Put back {len(put_back)} slices, Remaining slices:", len(img_slices))

    return selections


@dataclass
class WordProcessor:
    core: Callable[[NDArray], Tuple[List[int], Any]]

    def __call__(self, w: Word):
        w.labels, w.meta = self.core(w.img)
        return w

def process_words_stream(wp: WordProcessor, stream: Iterable[Word]) -> Generator[Word, None, None]:
    for w in stream:
        yield wp(w)

@dataclass
class WordProcCore0:
    model: Any
    window_tx: Callable[[NDArray], NDArray]
    meta_keys = ['conf', 'meta.avg_recons', 'bounds']

    def __call__(self, X: NDArray) -> Tuple[List[int], Any]:
        ret = get_chars2(X, self.model, self.window_tx)
        pred_y = [r['class'] for r in ret]
        # pred_names = [CLASS_NAMES[y] for y in pred_y]
        return pred_y, [sel_keys(r, self.meta_keys) for r in ret]



@dataclass
class Line:
    lineno: int
    words: List[Word] = field(default_factory=list)

    def decode(self, decoder):
        return [w.decode(decoder) for w in self.words]


@dataclass
class Page:
    pagename: str
    pageno: int
    lines: List[Line] = field(default_factory=list)

    def decode(self, decoder):
        return [l.decode(decoder) for l in self.lines]



# pX = np.random.rand(20,10)
# fig, (ax1,ax2) = plt.subplots(1,2)
# ax1.imshow(pX)
# ax2.imshow(square_pad(pX))

# ---- m
if __name__ == "__main__":
    wll = WordLoader(Path("./data/all_pages/"),
                     load_tx=partial(vizutils.resize_height, new_height=84))

    words = list(islice(wll, 20))
    wp = WordProcessor(core=WordProcCore0(model=model, window_tx=clf_tx))
    out_words = list(process_words_stream(wp, words))

    _pages = []

    for (pagename, pageno), page_group in groupby(out_words, key=lambda w: (w.pagename, w.pageno)):
        page = Page(pagename=pagename, pageno=pageno)

        for lineno, line_group in groupby(page_group, key=lambda w: w.lineno):
            line = Line(lineno=lineno)
            for w in line_group:
                line.words.append(w)
            page.lines.append(line)
        _pages.append(page)

    first_page = _pages[0]
    pprint(first_page.decode(lambda y: dss_charset.y_to_name[y]))

