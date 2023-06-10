import lightning.pytorch as pl
import torch as th
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        inp_dim=(1, 28, 28),
        num_classes=27,
        cls_scale: float = 0.5,
        rec_scale: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cls_scale = cls_scale
        self.rec_scale = rec_scale
        convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
        )

        with th.no_grad():
            out = convs(th.zeros((1, *inp_dim))).shape

        linear = nn.Sequential(
            nn.Linear(out[1], 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            # nn.LogSoftmax(),
        )

        self.encoder = nn.Sequential(convs, linear)
        self.softmax = nn.LogSoftmax()

        linear = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.GELU(),
            nn.Linear(128, out[1]),
            nn.GELU(),
            nn.Dropout(0.5),
        )

        deconvs = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(64, 12, 12)),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(64, 32, 3, 1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 3, 1),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(linear, deconvs)

        self.latent_loss = nn.CrossEntropyLoss()
        self.recon_loss = nn.MSELoss()
        # metric trackers
        self.t_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.v_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.mdrop = []

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters())
        return optimizer

    def step(self, batch, batch_idx, mode: str):
        x, y = batch

        latent = self.encoder(x)

        out = self.decoder(latent)

        probs = self.softmax(latent)

        lloss = self.latent_loss(probs, y) * self.cls_scale
        rloss = self.recon_loss(out, x) * self.rec_scale

        loss = lloss + rloss

        self.log(f"{mode}/latent_loss", lloss)
        self.log(f"{mode}/recon_loss", rloss)
        self.log(f"{mode}/loss", loss)
        return (
            loss,
            probs,
            out,
        )

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")[0]

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, probs, _ = self.step(batch, batch_idx, "test")

        # FIXME: should this not be out.argmaxed? or thoes the metric do that?
        # seems not
        # print("---------\n", out, y)
        acc = self.t_metric(probs, y)
        self.log("test/acc", acc)

    def validation_step(self, batch, batch_idx):
        _, out, _ = self.step(batch, batch_idx, "val")
        x, y = batch

        acc = self.v_metric(out, y)
        self.log("val/acc", acc)

    def enable_dropout(self):
        if self.mdrop:
            for m in self.mdrop:
                m.train()
        else:
            for m in self.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    self.mdrop.append(m)
                    m.train()

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return (self.softmax(latent), out)

    def uq_forward(self, x):
        self.enable_dropout()
        return self.forward(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        probs, out = self(x)
        return probs.argmax(axis=1)
