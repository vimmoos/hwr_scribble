from dataclasses import dataclass, field
from numpy.typing import NDArray
from typing import List
from benedict import BeneDict


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
