from dataclasses import dataclass
import cv2
from pathlib import Path
from typing import Optional, Callable
from numpy.typing import NDArray


@dataclass
class PageLoader:
    from_dir: Path
    glob: str = "*-binarized.jpg"
    imread_mode: int = cv2.IMREAD_GRAYSCALE

    load_tx: Optional[Callable[[NDArray], NDArray]] = None

    def list_paths(self):
        return list(sorted(self.from_dir.glob(self.glob)))

    def __len__(self):
        return len(self.list_paths())

    # @memoized_method(maxsize=100)
    def load_img(self, index):
        img = cv2.imread(str(self.list_paths()[index]),
                         self.imread_mode)
        return self.load_tx(img) if self.load_tx is not None else img

    def __getitem__(self, index):
        return self.load_img(index)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def iter_with_paths(self):
        path_list = self.list_paths()
        for i in range(len(self)):
            yield path_list[i], self[i]
