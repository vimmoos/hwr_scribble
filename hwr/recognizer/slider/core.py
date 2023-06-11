from hwr.recognizer.windows import WindowSlider
from typing import Iterable
import torch as th
from hwr.types import Word
from hwr.recognizer.slider.conf import RecognizerConf
from hwr.data_proc.char_proc import real_txs
from hwr.recognizer.slider.struct import RecognizerData
from hwr.utils.misc import batcher, to_tensor


def recognize_word(
    word: Word,
    conf: RecognizerConf,
):
    data = RecognizerData()
    with WindowSlider(
        width=conf.width,
        img=word.img,
        img_tx=real_txs,
        last_slice_accept=conf.last_slice,
    ) as window:
        while True:
            print("new window")
            print(window.pos)
            _confs, meta = yield window()
            print(_confs)
            if _confs is None:
                raise Exception("Mona ti non mi")

            data.update(_confs, window.freeze(), meta)

            window << conf.offset

            if data.max >= conf.perfect_conf:
                print("found perfect match ===============")
                width, pos = data.next_char()
                window.load(width, pos) << conf.shift(width, pos)

            if data.patience > conf.patience:
                print("done with patience ===============")
                width, pos = data.accept_recognition(conf.acceptable_conf)
                window.load(width, pos) << conf.shift(width, pos)

    _, _ = data.accept_recognition(conf.acceptable_conf)
    word.labels = data.labels
    word.meta = data.labels_meta
    yield word
    return


def recognize_words(
    words: Iterable[Word],
    conf: RecognizerConf,
    model: callable,
    batch_size: int = 64,
):
    batches = batcher(words, batch_size)
    for batch in batches:
        res = []
        recogs = [recognize_word(word, conf) for word in batch]
        windows = th.stack([to_tensor(next(rec)) for rec in recogs])
        while recogs:
            confs, probs, imgs = model(windows)
            next_windows = []
            next_recogs = []
            for idx, recog in enumerate(recogs):
                next_wind = recog.send(
                    (confs[idx], {"probs": probs[idx], "imgs": imgs[idx]})
                )
                if isinstance(next_wind, Word):
                    res.append(next_wind)
                    continue
                next_windows.append(to_tensor(next_wind))
                next_recogs.append(recog)

            windows = th.stack(next_windows)
            recogs = next_recogs
        yield sorted(res, key=Word.sort_key)
