import numpy as np
from contextlib import contextmanager
from collections import Counter


@contextmanager
def uncertainty(model, n=200, squeeze=False):
    def __inner__(data):
        nonlocal model, n
        probs, imgs = [], []
        for _ in range(n):
            prob, img = model.uq_forward(data)
            if prob.is_cuda:
                prob = prob.cpu()
                img = img.cpu()
            probs.append(prob.detach().numpy())
            imgs.append(img.detach().numpy().squeeze(axis=1))
        probs = np.moveaxis(np.array(probs), [0], [1])
        imgs = np.moveaxis(np.array(imgs), [0], [1])
        if squeeze:
            return probs.squeeze(), imgs.squeeze()
        return probs, imgs

    yield __inner__


@contextmanager
def confidences(model, only_confidence=True, n=200):
    def __inner__(data):
        nonlocal model, n
        with uncertainty(model, n=n) as m:
            probs, imgs = m(data)
            confs = [
                {k: v / probs.shape[1] for k, v in Counter(x).items()}
                for x in probs.argmax(axis=2)
            ]
            return confs if only_confidence else (confs, probs, imgs)

    yield __inner__


def wrap_confidence(model, n=200):
    def __inner__(data):
        with confidences(model, only_confidence=False, n=n) as m:
            confs, probs, imgs = m(data)
        return confs, probs, imgs.squeeze()

    return __inner__


def avg_uq(model, data, n=200):
    probs, imgs = np.array([]), np.array([])
    with uncertainty(model, squeeze=True) as m:
        probs, imgs = m(data)
    pavg, pstd = np.mean(probs, axis=0), np.std(probs, axis=0)
    return (
        (pavg, pstd),
        np.mean(imgs, axis=0),
    )
