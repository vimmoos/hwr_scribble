from torchvision.datasets import VisionDataset, ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np
from hwr.network.conf import BasicTrainEvalTask
from lightning.pytorch.loggers import WandbLogger
import torch
from typing import Any, Dict
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torch as th
import wandb
from pathlib import Path
from hwr.utils.misc import get_all_labels


def train_val_test_split(
    dataset: VisionDataset,
    test_size: float = 0.3,
    val_size_of_train: float = 0.2,
    # stratify
    stratify: bool = False,
    # randomness
    shuffle: bool = False,
    random_state: int = 42,
):
    """Three-way split for train/val/test set.

    Note if stratify is True then the y labels are used for stratification.

    Returns: a dict with train/val/test keys where values
             are `Subset` of the given `VisionDataset`.
    """

    # common kwargs
    base_kwargs = dict(
        random_state=random_state,
        shuffle=shuffle,
    )

    # apply strat if needed.
    if stratify:
        base_kwargs["stratify"] = get_all_labels(dataset)

    # compute first index split
    train_val_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=test_size,
        **base_kwargs,
    )

    # keep out the test set
    test_dataset = Subset(dataset, test_idx)
    _train_val_dataset = Subset(dataset, train_val_idx)  # the rest

    # reapply strat if needed from the remaining samples
    if stratify:
        base_kwargs["stratify"] = get_all_labels(_train_val_dataset)

    # compute second split
    train_idx, val_idx = train_test_split(
        np.arange(len(_train_val_dataset)),
        test_size=val_size_of_train,
        **base_kwargs,
    )

    # create the other subsets
    train_dataset = Subset(_train_val_dataset, train_idx)
    val_dataset = Subset(_train_val_dataset, val_idx)

    # return dataset dict
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def get_splits(conf: BasicTrainEvalTask):
    dataset = ImageFolder(str(conf.from_dir), transform=conf.tx)

    splits = train_val_test_split(
        dataset,
        test_size=conf.test_size,
        val_size_of_train=conf.val_size_of_train,
        random_state=conf.random_state,
        stratify=conf.stratify,
        shuffle=conf.shuffle,
    )

    print(
        "train/val/test:",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    return splits


def build_trainer(conf: BasicTrainEvalTask):
    wandb_logger = WandbLogger(
        project=conf.wandb_project,
        log_model=conf.log_model,
    )

    trainer_kwargs: Dict[str, Any] = dict(
        max_epochs=conf.max_epochs,
        logger=wandb_logger,
        callbacks=conf.callbacks or [],
    )

    trainer = pl.Trainer(**trainer_kwargs)
    return trainer


def build_dataset(conf: BasicTrainEvalTask):
    splits = get_splits(conf)
    input_shape = splits["train"][0][0].shape

    return splits, input_shape


def build_dataloaders(conf: BasicTrainEvalTask, splits):
    return {
        k: DataLoader(v, batch_size=conf.batch_size, num_workers=2)
        for k, v in splits.items()
    }


def build_model(conf: BasicTrainEvalTask, input_shape: int):
    # TODO: here should be a factory with signature **kwargs with at
    # least support for input_dim and num_classes
    return conf.model_cls(
        inp_dim=input_shape,
        num_classes=conf.num_classes,
    )


def train_and_eval_model(conf: BasicTrainEvalTask):
    trainer = build_trainer(conf)
    splits, input_shape = build_dataset(conf)
    dataloaders = build_dataloaders(conf, splits)
    model = build_model(conf, input_shape)

    print(model)
    # 3. Fitting
    trainer.fit(
        model=model,
        # Data
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
    )

    print("----- Train done ----")

    # 4. Testing on unseen
    trainer.test(dataloaders=dataloaders["test"])

    print("---- Test done ----")
    # results are reported on wandb
    #
    if conf.return_test_set_predictions:
        return (
            model,
            get_all_labels(dataloaders["test"].dataset),
            th.cat(
                trainer.predict(
                    model,
                    dataloaders=dataloaders["test"],
                )
            ),
        )

    return model, None, None

def build_from_checkpoint(model, path):
    if torch.cuda.is_available():
        return model.load_from_checkpoint(path)
    return model.load_from_checkpoint(path, map_location=torch.device("cpu"))



def load_model(model, model_str="drl42/HWR_UQ/model-3387c7fi:v0"):
    path = Path("artifacts") / Path(model_str).stem / "model.ckpt"
    if path.is_file():
        return build_from_checkpoint(model, path)

    run = wandb.init()
    artifact = run.use_artifact(model_str, type="model")
    _ = artifact.download()

    wandb.finish()
    return build_from_checkpoint(model, path)
