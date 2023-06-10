import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any, Tuple
from dataclasses import field, dataclass


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
