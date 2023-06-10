from dataclasses import dataclass
from recognizer.windows import SliceAcceptor, accept_slices_bounce


@dataclass
class RecognizerConf:
    width: int = 40
    last_slice: SliceAcceptor = accept_slices_bounce
    patience: int = 3
    offset: int = 20
    perfect_conf: float = 0.95
    acceptable_conf: float = 0.80
    shift: callable = lambda width, offset: width
