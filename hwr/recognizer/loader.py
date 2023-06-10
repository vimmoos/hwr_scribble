from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from numpy.typing import NDArray
from hwr.recognizer.struct import Word
import cv2


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
                    yield Word(
                        pagename=page_path.name,
                        pageno=pageno,
                        lineno=lino,
                        wordno=wordno,
                        img=img,
                    )

    def get_one_word(self, pageno, lino, wordno):
        return self.load_img(self.list_word_paths(pageno, lino)[wordno])
