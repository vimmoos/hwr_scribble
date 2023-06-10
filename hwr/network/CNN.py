import lightning.pytorch as pl
import torch as th
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy


class CNN(pl.LightningModule):
    """
    Simple CNN architecture for Character classification
    """

    def __init__(self, inp_dim=(1, 28, 28), num_classes=27):
        super().__init__()
        self.save_hyperparameters()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
        )

        with th.no_grad():
            out = self.convs(th.zeros((1, *inp_dim))).shape

        self.linear = nn.Sequential(
            nn.Linear(out[1], 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(),
        )
        self.loss = nn.CrossEntropyLoss()
        # metric trackers
        self.t_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.v_metric = Accuracy(task="multiclass", num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters())
        return optimizer

    def step(self, batch, batch_idx, mode: str):
        x, y = batch

        out = self.linear(self.convs(x))
        loss = self.loss(out, y)

        self.log(f"{mode}/loss", loss)
        return loss, out

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")[0]

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, out = self.step(batch, batch_idx, "test")

        acc = self.t_metric(out, y)
        self.log("test/acc", acc)

    def validation_step(self, batch, batch_idx):
        _, out = self.step(batch, batch_idx, "val")
        x, y = batch

        acc = self.v_metric(out, y)
        self.log("val/acc", acc)

    def forward(self, x):
        fmaps = self.convs(x)
        return self.linear(fmaps)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self(x).argmax(axis=1)
