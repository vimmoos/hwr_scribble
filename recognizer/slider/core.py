from hwr2.tasks.uq.windows import (
    WindowSlider,
    accept_slices_bounce,
    SliceAcceptor,
)
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any, Tuple, Iterable
from dataclasses import field, dataclass
from benedict import BeneDict
import torch as th


import cv2
from hwr2.tasks.segment_lines import transforms as lin_seg_tx


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


@dataclass
class RecognizerConf:
    width: int = 40
    last_slice: SliceAcceptor = accept_slices_bounce
    patience: int = 3
    offset: int = 20
    perfect_conf: float = 0.95
    acceptable_conf: float = 0.80
    shift: callable = lambda width, offset: width


@dataclass
class RecognizerData:
    max: float = -np.inf
    patience: int = 0

    meta: Any = None
    confs: Dict[Any, Any] = field(default_factory=dict)
    window: Tuple[int, int, NDArray] = field(default_factory=tuple)
    labels: List[int] = field(default_factory=list)
    labels_meta: List[Any] = field(default_factory=list)

    def _update(self, confs=None, window_freeze=None, meta=None):
        self.confs = confs or dict()
        self.patience = 0
        self.max = -np.inf if confs is None else max(confs.values())
        self.window = window_freeze or tuple()
        self.meta = meta

    def is_better(self, confs):
        return max(confs.values()) > self.max

    def update(self, confs, window_freeze, meta):
        if self.is_better(confs):
            self._update(confs, window_freeze, meta)
        else:
            self.patience += 1

    def next_char(self):
        self.labels.append(max(self.confs, key=self.confs.get))
        self.labels_meta.append(
            {
                "confs": self.confs,
                "window": self.window,
                "meta_model": self.meta,
            }
        )
        width, pos = self.window[0], self.window[1]
        self._update()
        return width, pos

    def accept_recognition(self, threshold):
        if self.max < threshold:
            width, pos = self.window[0], self.window[1]
            self._update()
            return width, pos
        else:
            return self.next_char()


def recognize_word(
    word: Word,
    conf: RecognizerConf,
):
    data = RecognizerData()
    with WindowSlider(
        width=conf.width,
        img=word.img,
        img_tx=clf_tx,
        last_slice_accept=conf.last_slice,
    ) as window:
        while True:
            _confs, meta = yield window()
            if _confs is None:
                raise Exception("Mona ti non mi")

            data.update(_confs, window.freeze(), meta)

            window << conf.offset

            if data.max >= conf.perfect_conf:
                width, pos = data.next_char()
                window.load(width, pos) << conf.shift(width, pos)

            if data.patience > conf.patience:
                width, pos = data.accept_recognition(conf.acceptable_conf)
                window.load(width, pos) << conf.shift(width, pos)

    _, _ = data.accept_recognition(conf.acceptable_conf)
    word.labels = data.labels
    word.meta = data.labels_meta
    yield word
    return


def batcher(
    iterable: Iterable,
    batch_size: int = 64,
    appender: callable = lambda acc, y: [y] if acc is None else acc + [y],
):
    batch = None
    for el in iterable:
        batch = appender(batch, el)
        if len(batch) == batch_size:
            yield batch
            batch = None
    if batch is not None:
        yield batch


def to_tensor(img: NDArray):
    return th.Tensor(img / 255).unsqueeze(0)


def recognize_words(
    words: Iterable[Word],
    conf: RecognizerConf,
    model: callable,
    batch_size: int = 64,
):
    batches = batcher(words, batch_size)
    for batch in batches:
        res = []
        recogs = [recognize_word(word, conf) for word in batch]
        windows = th.stack([to_tensor(next(rec)) for rec in recogs])
        while recogs:
            confs, probs, imgs = model(windows)
            next_windows = []
            next_recogs = []
            for idx, recog in enumerate(recogs):
                next_wind = recog.send(
                    (confs[idx], {"probs": probs[idx], "imgs": imgs[idx]})
                )
                if isinstance(next_wind, Word):
                    res.append(next_wind)
                    continue
                next_windows.append(to_tensor(next_wind))
                next_recogs.append(recog)

            windows = th.stack(next_windows)
            recogs = next_recogs
        yield sorted(res, key=Word.sort_key)
