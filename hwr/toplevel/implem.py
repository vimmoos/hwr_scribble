from itertools import groupby
from typing import Iterable, Generator

from hwr import Word, Page, Line


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


from dataclasses import dataclass
from typing import Callable, Tuple, List, Any
from numpy.typing import NDArray
from hwr.utils.misc import sel_keys
from hwr.recognizer.peaker.core import get_chars2


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
