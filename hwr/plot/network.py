from hwr.utils.misc import shift_in_unit
import numpy as np
from hwr.network.uq.core import confidences
import matplotlib.pyplot as plt
from scipy.stats import entropy


def predict_single(model, data, y):
    probs, imgs = None, None
    with confidences(model, only_confidence=False) as m:
        confs, probs, imgs = m(data)

    confs = confs[0]
    probs = probs.squeeze()
    imgs = imgs.squeeze()

    pavg, pstd = np.mean(probs, axis=0), np.std(probs, axis=0)
    img, simg = (np.mean(imgs, axis=0), np.std(imgs, axis=0))
    max_pavg = pavg.max()
    probs = shift_in_unit(pavg)
    fig, ((a1, a2), (a3, a4), (a5, a6)) = plt.subplots(3, 2, figsize=(10, 10))
    fig.suptitle(f"Highest prob={np.argmax(pavg,axis=0)}\n correct= {y}")
    a1.set_title(f"log softmax probs, max={max_pavg:.2f}")
    a1.barh(y=range(27), width=probs, xerr=shift_in_unit(pstd) / 2)
    a1.set_yticks(range(27))
    a2.set_title("Confidence")
    a2.barh(y=[str(x) for x in list(confs.keys())], width=list(confs.values()))
    for i, v in enumerate(confs.values()):
        a2.text(0.5, i, str(v), ha="left", va="center")
    # a2.yticks(np.arange(len(confs.keys())), confs.keys())
    # a2.set_yticks(list(confs.keys()))
    a3.set_title("reconstructed")
    a3.imshow(img)
    a4.set_title("original")
    if data.is_cuda:
        data = data.cpu()
    a4.imshow(data.squeeze())
    a5.set_title(f"deviation mean={np.mean(simg):.2f}")
    x = a5.imshow(simg)
    fig.colorbar(x)
    a6.set_axis_off()
    a6.text(
        0.2,
        0.2,
        f"image dev dev {np.std(simg)}\nimage deviation entropy {entropy(simg.flatten(),np.ones((28,28)).flatten()*np.mean(simg))}",
    )
    fig.tight_layout()
    return fig
