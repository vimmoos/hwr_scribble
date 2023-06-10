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


import matplotlib.pyplot as plt
import numpy as np
import torch as th
from benedict import BeneDict
from benedict import OrderedBeneDict
from hwr2.common import dss_charset
from hwr2.common import vizutils
from hwr2.tasks.uq import uq_utils
from hwr2.tasks.uq.autoencoder import Autoencoder
from hwr2.tasks.uq.utils import shift_in_unit
from numpy.typing import NDArray
from peakdetect import peakdetect


# run = wandb.init()
# artifact = run.use_artifact("drl42/HWR_UQ/model-3387c7fi:v0", type="model")
# artifact_dir = artifact.download()

# import os

# os.listdir(artifact_dir)

# checkpoint = Path(artifact_dir) / "model.ckpt"
# model = Autoencoder.load_from_checkpoint(checkpoint)

# model.cpu()


# def mk_slice_seq(slices, max_slices, slice_tx):
#     return [slice_tx(np.hstack(list(reversed(slices[0:n])))) for n in range(1,max_slices+1)]


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
    wll = WordLoader(
        Path("./data/all_pages/"),
        load_tx=partial(vizutils.resize_height, new_height=84),
    )

    words = list(islice(wll, 20))
    wp = WordProcessor(core=WordProcCore0(model=model, window_tx=clf_tx))
    out_words = list(process_words_stream(wp, words))

    _pages = []

    for (pagename, pageno), page_group in groupby(
        out_words, key=lambda w: (w.pagename, w.pageno)
    ):
        page = Page(pagename=pagename, pageno=pageno)

        for lineno, line_group in groupby(page_group, key=lambda w: w.lineno):
            line = Line(lineno=lineno)
            for w in line_group:
                line.words.append(w)
            page.lines.append(line)
        _pages.append(page)

    first_page = _pages[0]

    pprint(first_page.decode(lambda y: dss_charset.y_to_name[y]))
