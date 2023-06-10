import cv2
from network.uq import wrap_confidence
from network import load_model, Autoencoder
from recognizer.slider.core import recognize_word
from recognizer.struct import Word
from recognizer.slider.conf import RecognizerConf
from utils.misc import to_tensor
import matplotlib.pyplot as plt
import plot.utils as vz

img = cv2.imread(
    "/home/vimmoos/hwr-project/notebooks/simpified/all_pages/page-01.d/lines.d/line-01.d/words.d/word-3.png",
    cv2.IMREAD_GRAYSCALE,
)
model = load_model(Autoencoder).cpu()

cmodel = wrap_confidence(model, n=10)

gen = recognize_word(
    Word("mona", 1, 1, 1, img), RecognizerConf(patience=1, offset=10)
)

win = next(gen)
while True:
    confs, probs, imgs = cmodel(to_tensor(win).unsqueeze(0))
    win = gen.send((confs[0], imgs))
    if isinstance(win, Word):
        break

print(win.labels)


# plt.imshow(x.meta[0]["meta_model"])
plt.imshow(
    vz.make_hstack_image(reversed([m.get("window")[2] for m in win.meta]))
)
plt.show()
