from hwr import Word
from typing import Iterable, Generator
from hwr.network import (
    Autoencoder,
    model_aug,
    load_model,
    model_aug_91,
    model_aug_82,
    model_van,
)
from hwr.network.uq import wrap_confidence, ensemble_wrap_confidence
from hwr.recognizer.slider import (
    recognize_word,
    RecognizerConf,
    recognize_words,
)
from hwr.utils.misc import to_tensor


def mona(words: Iterable[Word]) -> Generator[Word, None, None]:
    return recognize_words(
        words,
        RecognizerConf(),
        ensemble_wrap_confidence(
            [
                load_model(Autoencoder, model_aug).cuda(),
                load_model(Autoencoder, model_van).cuda(),
                load_model(Autoencoder, model_aug_82).cuda(),
                load_model(Autoencoder, model_aug_91).cuda(),
            ],
            n=1000,
        ),
        128,
    )


# def mona(words: Iterable[Word]) -> Generator[Word, None, None]:
#     return recognize_words(
#         words,
#         RecognizerConf(),
#         wrap_confidence(load_model(Autoencoder, model_aug).cuda(), n=1000),
#         128,
#     )


# def mona(words: Iterable[Word]) -> Generator[Word, None, None]:
#     print("mona", words)
#     model = wrap_confidence(load_model(Autoencoder, model_aug).cuda(), n=200)
#     for word in words:
#         print("mona")
#         gen = recognize_word(word, RecognizerConf())
#         win = next(gen)
#         while True:
#             confs, probs, imgs = model(to_tensor(win).unsqueeze(0).cuda())
#             win = gen.send((confs[0], imgs))
#             if isinstance(win, Word):
#                 break
#         yield word
