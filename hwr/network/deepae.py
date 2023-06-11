import lightning.pytorch as pl
import torch as th
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from typing import Callable, List, Optional


def conv_block(
    in_channels: int,
    out_channels: int,
    closing_ops: Callable[[], List[nn.Module]] = lambda in_ch, out_ch: [],
    act_fun: Callable[[], nn.Module] = lambda: nn.GELU(),
    out_chs_firstconv: Optional[int] = None,
    out_chs_secondconv: Optional[int] = None,
    in_chs_firstconv: Optional[int] = None,
    in_chs_secondconv: Optional[int] = None,
    in_chs_closing: Optional[int] = None,
):
    out_chs_firstconv = out_chs_firstconv or in_channels
    out_chs_secondconv = out_chs_secondconv or out_channels
    in_chs_secondconv = in_chs_secondconv or in_channels
    in_chs_closing = in_chs_closing or in_channels
    in_chs_firstconv = in_chs_firstconv or in_channels

    return nn.Sequential(
        nn.Conv2d(
            in_chs_firstconv,
            out_chs_firstconv,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        act_fun(),
        nn.Conv2d(
            in_chs_secondconv,
            out_chs_secondconv,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        act_fun(),
        *closing_ops(in_chs_closing, out_channels),
    )


class DeepAutoencoder(pl.LightningModule):
    def __init__(
        self,
        inp_dim=(1, 28, 28),
        num_classes=27,
        cls_scale: float = 0.5,
        rec_scale: float = 0.5,
        n_filters: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cls_scale = cls_scale
        self.rec_scale = rec_scale
        self.start = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.GELU(),
        )
        self.bconv1 = conv_block(n_filters, n_filters)
        self.bconv2 = conv_block(2 * n_filters, 2 * n_filters)
        self.bconv3 = conv_block(3 * n_filters, 3 * n_filters)
        self.flatten = nn.Sequential(
            nn.Conv2d(3 * n_filters, 3 * n_filters, 3, 1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
        )

        with th.no_grad():
            inp = th.zeros((1, *inp_dim))
            sout = self.start(inp)
            out = self.bconv1(sout)
            sout = self.bconv2(th.cat([sout, out], dim=1))
            out = self.bconv3(th.cat([out, sout], dim=1))
            out = self.flatten(out).shape

        self.enc_linear = nn.Sequential(
            nn.Linear(out[1], 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        self.softmax = nn.LogSoftmax()

        self.dec_linear = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.GELU(),
            nn.Linear(128, out[1]),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Unflatten(dim=1, unflattened_size=(3 * n_filters, 13, 13)),
            nn.UpsamplingNearest2d(scale_factor=2),
        )

        self.bdeconv1 = conv_block(
            2 * n_filters, n_filters, in_chs_firstconv=3 * n_filters
        )

        self.bdeconv2 = conv_block(
            2 * n_filters, n_filters, in_chs_firstconv=4 * n_filters
        )
        self.bdeconv3 = conv_block(
            n_filters, n_filters, in_chs_firstconv=2 * n_filters
        )

        self.bdeconv4 = conv_block(
            n_filters,
            1,
            in_chs_firstconv=n_filters * 4,
            out_chs_firstconv=n_filters * 3,
            in_chs_secondconv=n_filters * 3,
            out_chs_secondconv=n_filters,
            closing_ops=lambda in_ch, out_ch: [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1),
                nn.Tanh(),
            ],
        )

        self.latent_loss = nn.CrossEntropyLoss()
        self.recon_loss = nn.MSELoss()
        # metric trackers
        self.t_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.v_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.mdrop = []

    def encoder(self, inp):
        sout = self.start(inp)
        out = self.bconv1(sout)
        sout = self.bconv2(th.cat([sout, out], dim=1))
        out = self.bconv3(th.cat([out, sout], dim=1))
        out = self.flatten(out)
        return self.enc_linear(out)

    def decoder(self, inp):
        inp = self.dec_linear(inp)
        out = self.bdeconv1(inp)
        sout = self.bdeconv2(th.cat([inp, out], dim=1))
        out = self.bdeconv3(th.cat([out, sout], dim=1))
        return self.bdeconv4(th.cat([inp, out], dim=1))


class DeepAutoencoder2(pl.LightningModule):
    def __init__(
        self,
        inp_dim=(1, 28, 28),
        num_classes=27,
        cls_scale: float = 0.5,
        rec_scale: float = 0.5,
        n_filters: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cls_scale = cls_scale
        self.rec_scale = rec_scale
        self.start = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.GELU(),
        )
        self.bconv1 = conv_block(n_filters, n_filters)
        self.bconv2 = conv_block(2 * n_filters, 2 * n_filters)
        self.bconv3 = conv_block(3 * n_filters, 3 * n_filters)
        self.flatten = nn.Sequential(
            nn.Conv2d(3 * n_filters, 3 * n_filters, 3, 1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
        )

        with th.no_grad():
            inp = th.zeros((1, *inp_dim))
            sout = self.start(inp)
            out = self.bconv1(sout)
            sout = self.bconv2(th.cat([sout, out], dim=1))
            out = self.bconv3(nn.MaxPool2d(2)(th.cat([out, sout], dim=1)))
            out = self.flatten(out).shape
            # print(out.shape)
            # out = nn.Flatten()(out).shape

        self.enc_linear = nn.Sequential(
            nn.Linear(out[1], 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        self.softmax = nn.LogSoftmax()

        self.dec_linear = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.GELU(),
            nn.Linear(128, out[1]),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Unflatten(dim=1, unflattened_size=(3 * n_filters, 6, 6)),
            nn.UpsamplingNearest2d(scale_factor=2),
        )

        self.bdeconv1 = conv_block(
            2 * n_filters, n_filters, in_chs_firstconv=3 * n_filters
        )

        self.bdeconv2 = conv_block(
            2 * n_filters, n_filters, in_chs_firstconv=4 * n_filters
        )
        self.bdeconv3 = conv_block(
            n_filters, n_filters, in_chs_firstconv=2 * n_filters
        )

        self.bdeconv4 = conv_block(
            n_filters,
            1,
            in_chs_firstconv=n_filters * 4,
            out_chs_firstconv=n_filters * 3,
            in_chs_secondconv=n_filters * 3,
            out_chs_secondconv=n_filters,
            closing_ops=lambda in_ch, out_ch: [
                nn.ConvTranspose2d(in_ch, in_ch, kernel_size=3, stride=1),
                nn.GELU(),
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1),
                nn.Tanh(),
            ],
        )

        self.upsamp = nn.UpsamplingNearest2d(scale_factor=2)
        self.max_pool = nn.MaxPool2d(2)

        self.latent_loss = nn.CrossEntropyLoss()
        self.recon_loss = nn.MSELoss()
        # metric trackers
        self.t_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.v_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.mdrop = []

    def encoder(self, inp):
        sout = self.start(inp)
        out = self.bconv1(sout)
        sout = self.bconv2(th.cat([sout, out], dim=1))
        out = self.bconv3(self.max_pool(th.cat([out, sout], dim=1)))
        out = self.flatten(out)
        return self.enc_linear(out)

    def decoder(self, inp):
        inp = self.dec_linear(inp)
        out = self.bdeconv1(inp)
        sout = self.bdeconv2(th.cat([inp, out], dim=1))

        out = self.bdeconv3(self.upsamp(th.cat([out, sout], dim=1)))
        return self.bdeconv4(th.cat([self.upsamp(inp), out], dim=1))
