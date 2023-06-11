import logging
import shutil
from dataclasses import dataclass
from itertools import groupby
from logging import Logger
from os.path import splitext
from pathlib import Path
from typing import Callable, Tuple, List, Any
from typing import Iterable, Generator

from numpy.typing import NDArray

from hwr import Word, Page, Line
from hwr.recognizer.peaker.core import get_chars2
from hwr.utils import dss_charset
from hwr.utils.misc import sel_keys


def word_stream_to_pages(processed_stream: Iterable[Word]) -> Generator[Page, None, None]:
    """Convert a stream of Word objects to a stream of Page objects.

    NOTE: !!! the stream of words must have monotonically increasing pageno/lineno/wordno !!!
    """

    # performs a streaming group-by-pageno (we also take the name to include that as well
    # the number of groups should be unchanged, i.e. 1<->1 pagename<->pageno)
    for (pagename, pageno), page_group in groupby(
            processed_stream, key=lambda w: (w.pagename, w.pageno)
    ):
        # build page for this group
        yield Page(
            pagename=pagename,
            pageno=pageno,
            lines=[Line(lineno=lineno, words=[w for w in line_group])
                   for (lineno, line_group)
                   in groupby(page_group, key=lambda w: w.lineno)]
        )


@dataclass
class SimpleWordStreamProc:
    core: Callable[[NDArray], Tuple[List[int], Any]]

    def __call__(self, stream: Iterable[Word]) -> Generator[Word, None, None]:
        for word in stream:
            word.labels, word.meta = self.core(word.img)
            yield word


@dataclass
class WordProcCore0:
    """An implementation of word processor core using the custom projection based windows approach"""
    model: Any
    window_tx: Callable[[NDArray], NDArray]
    torch_device: str = "cpu"
    meta_keys = ["conf", "meta.avg_recons", "bounds"]

    def __call__(self, X: NDArray) -> Tuple[List[int], Any]:
        ret = get_chars2(X, self.model, self.window_tx, torch_device=self.torch_device)
        pred_y = [r["class"] for r in ret]
        return pred_y, [sel_keys(r, self.meta_keys) for r in ret]


def write_page(page: Page, into_dir: Path, suffix="_characters.txt"):
    page_name, _ = splitext(page.pagename)
    dest = into_dir / f"{page_name}{suffix}"
    with open(dest, "wb") as fo:
        fo.write(page.to_string(dss_charset.y_to_unicode.__getitem__).encode("utf-8"))
    return dest


@dataclass
class UnicodePageWriter:
    log: Logger
    rm_dest: bool = True

    def __call__(self, out_dir: Path, page_stream: Iterable[Page]) -> None:
        if self.rm_dest and out_dir.exists():
            self.log.warning("Deleting old contents")
            shutil.rmtree(out_dir)

        out_dir.mkdir(exist_ok=False)

        for page in page_stream:
            self.log.info("Wrting page: %s", page.pagename)
            write_page(page, out_dir)
