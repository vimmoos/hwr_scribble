from dataclasses import dataclass
from hwr.recognizer.windows import SliceAcceptor, accept_slices_bounce


@dataclass
class RecognizerConf:
    width: int = 30
    last_slice: SliceAcceptor = accept_slices_bounce
    patience: int = 1
    offset: int = 5
    perfect_conf: float = 0.95
    acceptable_conf: float = 0.8
    shift: callable = lambda width, offset: width
