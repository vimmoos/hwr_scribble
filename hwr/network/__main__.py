from sklearn.metrics import classification_report
import wandb
from hwr.network import conf, utils, autoencoder

from pathlib import Path
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
from torchsummary import summary
import numpy as np

torch.set_float32_matmul_precision("medium")
# set this to true to save trained checkpoints
# WARN: slows down the final step of wandb sync, so use false in debugging/test
# scenarios

# LOG_MODEL = False
LOG_MODEL = False
WANDB_PROJECT = "HWR_UQ"

run = wandb.init(project=WANDB_PROJECT)

train_task_config = conf.BasicTrainEvalTask(
    from_dir=Path("data/dss/monkbrill-prep-tri"),
    model_cls=autoencoder.Autoencoder,
    tx=conf.txs,
    num_classes=27,
    stratify=True,
    random_state=42,
    shuffle=True,
    batch_size=64,
    wandb_project=WANDB_PROJECT,
    max_epochs=2000,
    log_model=LOG_MODEL,
    callbacks=[
        EarlyStopping(
            monitor="val/latent_loss",
            patience=5,
        )
    ],
)

model, y_true, y_preds = utils.train_and_eval_model(train_task_config)
print(summary(model, input_size=(1, 28, 28), device="cpu"))
y_preds = np.array(y_preds)
print(classification_report(y_true, y_preds))

wandb.finish()
