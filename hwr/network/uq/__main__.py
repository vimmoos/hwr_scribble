from hwr.network import Autoencoder, load_model
from hwr.network import conf
from hwr.plot.network import predict_single

from torchvision.datasets import ImageFolder
from pathlib import Path

autoencoder = load_model(Autoencoder)
dataset = ImageFolder(Path("data/dss/monkbrill-prep-tri"), transform=conf.txs)


test = dataset[0][0].unsqueeze(1).cuda()
test_y = dataset[0][1]
fig = predict_single(autoencoder, test, test_y)
fig.show()
fig.waitforbuttonpress()
