from hwr import Word
from typing import Iterable, Generator
from hwr.network import Autoencoder, model_aug
from hwr.network.uq import wrap_confidence
from hwr.recognizer.slider import recognize_word, RecognizerConf
from hwr.utils.misc import to_tensor


def mona(words: Iterable[Word]) -> Generator[Word, None, None]:
    model = wrap_confidence(load_model(Autoencoder, model_aug).cuda(), n=200)
    for word in words:
        gen = recognize_word(word, RecognizerConf())
        win = next(gen)
        while True:
            confs, probs, imgs = model(to_tensor(win).unsqueeze(0).cuda())
            win = gen.send((confs[0], imgs))
            if isinstance(win, Word):
                break
        yield word
