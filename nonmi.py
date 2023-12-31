from dataclasses import dataclass
from functools import partial
from itertools import groupby
from itertools import islice
from os.path import splitext
from pathlib import Path
from typing import Callable, Tuple, Any, Generator, Iterable
from typing import List

from numpy.typing import NDArray

from hwr import Line, Page
from hwr.data_proc.utils import clf_tx
from hwr.network import load_model, Autoencoder
from hwr.plot import utils as vizutils
from hwr.recognizer.loader import WordLoader
from hwr.recognizer.peaker.core import get_chars2
from hwr.types import Word
from hwr.utils import dss_charset
from hwr.utils.misc import sel_keys

WordsPager = Callable[[Iterable[Word]], Generator[Page, None, None]]

# --- Word stream transformer (word-stream -> word-stream)---
WordStreamProcessor = Callable[[Iterable[Word]], Generator[Word, None, None]]


@dataclass
class SimpleWordStreamProc:
    core: Callable[[NDArray], Tuple[List[int], Any]]

    def __call__(self, stream: Iterable[Word]) -> Generator[Word, None, None]:
        for word in stream:
            word.labels, word.meta = self.core(word.img)
            yield word


#  --- Word stream aggregator (word-stream -> page-stream)---
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


def limit_pages(word_stream: Iterable[Word], max_pages) -> Generator[Word, None, None]:
    """Return a generator that yields words from `word_stream` from the first `max_pages` encountered pages.
    """
    selected = islice(groupby(word_stream, lambda w: w.pageno), max_pages)
    for _, wordz in selected:
        yield from wordz


def write_page(page: Page, into_dir: Path):
    page_name, _ = splitext(page.pagename)
    dest = into_dir / f"{page_name}.txt"
    with open(dest, "wb") as fo:
        fo.write(page.to_string(dss_charset.y_to_unicode.__getitem__).encode("utf-8"))
    return dest


import shutil


@dataclass
class UnicodePageWriter:
    out_dir: Path
    rm_dest: bool = True

    def __call__(self, page_stream: Iterable[Page]) -> None:
        if self.rm_dest and self.out_dir.exists():
            print("Deleting old contents")
            shutil.rmtree(self.out_dir)

        self.out_dir.mkdir(exist_ok=False)

        for page in page_stream:
            print("Wrting page:", page.pagename)
            write_page(page, self.out_dir)


#  --- test for limit pages ---
# wll_test = WordLoader(
#     Path("./data/all_pages/"),
#     load_tx=partial(vizutils.resize_height, new_height=84),
# )
#
# by_page = groupby(limit_pages(wll_test, 3), key=lambda x: x.pageno)
#
# for k, page_words in by_page:
#     print(k, len(list(page_words)))


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


if __name__ == "__main__":
    DEVICE = "cuda"
    from pprint import pprint

    # 0-0.5 -> pages.d (diag_path)

    # 1. Create a word loader
    wll = WordLoader(
        # Path("./data/all_pages/"),
        Path("./data/debug_pages/"),
        load_tx=partial(vizutils.resize_height, new_height=84),
    )

    # 2. Instantiate the model
    model = load_model(Autoencoder)
    model.to(DEVICE)

    # 3. create a word-processror with the appropriate core
    wsp = SimpleWordStreamProc(core=WordProcCore0(model=model, window_tx=clf_tx, torch_device=DEVICE))

    # 4. pick a sample of words for the test
    words = limit_pages(wll, 1)

    # 5. create the stream of processed words
    processed_words_stream = wsp(words)
    # 6. Aggregate the word stream into pages
    page_stream = word_stream_to_pages(processed_words_stream)
    # force realization of all pages
    pages = list(page_stream)

    first_page = pages[0]

    pprint(first_page.decode(dss_charset.y_to_name.__getitem__))
    pprint(first_page.decode(dss_charset.y_to_unicode.__getitem__))
    print(first_page.to_string(dss_charset.y_to_unicode.__getitem__))

    pwriter = UnicodePageWriter(Path("./results"), rm_dest=True)
    pwriter(pages)
