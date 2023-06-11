from .autoencoder import Autoencoder
from .CNN import CNN
from .conf import BasicTrainEvalTask
from .utils import load_model, train_and_eval_model


model_aug = "drl42/HWR_UQ/model-sfkydqpd:v0"
model_van = "drl42/HWR_UQ/model-3387c7fi:v0"
model_aug_82 = "drl42/HWR_UQ/model-dzs749xm:v0"
model_aug_91 = "drl42/HWR_UQ/model-qme1us6k:v0"
model_deep = "drl42/HWR_UQ/model-u1x4055l:v0"
model_dfhyper = "drl42/HWR_UQ/model-33q7nqvc:v0"

__all__ = [
    "Autoencoder",
    "CNN",
    "BasicTrainEvalTask",
    "load_model",
    "train_and_eval_model",
    "model_aug",
    "model_van",
    "model_aug_82",
    "model_aug_91",
    "model_dfhyper",
    "model_deep",
]
