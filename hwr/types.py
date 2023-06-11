from dataclasses import dataclass, field
from typing import List
from benedict import BeneDict
from numpy.typing import NDArray


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
        return self.pageno, self.lineno, self.wordno

    def decode(self, decoder):
        return [decoder(y) for y in self.labels]

    def to_string(self, decoder, sep=" "):
        return sep.join(self.decode(decoder))


@dataclass
class Line:
    lineno: int
    words: List[Word] = field(default_factory=list)

    def decode(self, decoder):
        return [w.decode(decoder) for w in self.words]

    def to_string(self, decoder, sep=" ", char_sep=""):
        words = [w.to_string(decoder, sep=char_sep) for w in self.words]
        words = [w for w in words if w]  # trim nones and empty string
        return sep.join(words)


@dataclass
class Page:
    pagename: str
    pageno: int
    lines: List[Line] = field(default_factory=list)

    def decode(self, decoder):
        return [l.decode(decoder) for l in self.lines]

    def to_string(self, decoder, sep="\n", word_sep=" ", char_sep=""):
        lines = [l.to_string(decoder, sep=word_sep, char_sep=char_sep) for l in self.lines]
        lines = [l for l in lines if l]
        return sep.join(lines)


class AnomalyError(Exception):
    pass


class Div0RiskError(AnomalyError):
    pass
