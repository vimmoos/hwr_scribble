import shutil
from .core import PageProcessor
import logging
from pathlib import Path
from dataclasses import dataclass
import cv2


@dataclass
class PageAnalysisWriter:
    root: Path

    lines_dir = "lines.d"
    words_dir = "words.d"

    def write_lines(self, ppr: PageProcessor):
        lines_dir = self.root / self.lines_dir
        if lines_dir.exists():
            logging.warning("Deleting contents of: %s", lines_dir)
            shutil.rmtree(lines_dir)
        max_key = max(ppr.s.lino_img_index.keys())
        n_digits = len(str(max_key))

        for lino, img in ppr.s.lino_img_index.items():
            lino_fmt = str(lino).zfill(n_digits)
            lino_dir = lines_dir / f"line-{lino_fmt}.d"
            words_dir = lino_dir / self.words_dir
            words_dir.mkdir(parents=True)

            # write the line image
            line_dest = lino_dir / f"line.png"
            cv2.imwrite(str(line_dest), img)

            # write the individual words
            words = ppr.s.lino_word_imgs[lino]
            n_words = len(words)
            n_digits_2 = len(str(n_words))

            for wordno, word_img in enumerate(words):  # ppr.s.lino_word_imgs[lino].items():
                wordno_fmt = str(wordno).zfill(n_digits_2)
                word_dest = words_dir / f"word-{wordno_fmt}.png"
                cv2.imwrite(str(word_dest), word_img)
