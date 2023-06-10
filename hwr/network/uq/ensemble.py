import torch as th
from tqdm import trange
from hwr.network import BasicTrainEvalTask
from hwr.utils.misc import get_all_labels
from collections import Counter
from hwr.network.utils import (
    build_model,
    build_dataset,
    build_dataloaders,
)
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


def build_trainer(conf: BasicTrainEvalTask):
    trainer_kwargs = dict(
        max_epochs=conf.max_epochs,
        callbacks=[
            EarlyStopping(
                monitor="val/loss",
                patience=5,
            )
        ],
    )

    trainer = pl.Trainer(**trainer_kwargs)
    return trainer


def train_and_eval_ensemble(conf: BasicTrainEvalTask):
    trainers, models = [], []
    splits, input_shape = build_dataset(conf)

    for i in trange(conf.num_estimators):
        trainers.append(build_trainer(conf))
        models.append(build_model(conf, input_shape))
        dataloaders = build_dataloaders(conf, splits)
        # 3. Fitting
        trainers[i].fit(
            model=models[i],
            # Data
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"],
        )

    print("----- Train done ----")
    #
    if not conf.return_test_set_predictions:
        return None, None

    preds = []

    for i in range(conf.num_estimators):
        dataloaders = build_dataloaders(conf, splits)
        preds.append(
            list(
                th.cat(
                    trainers[i].predict(
                        models[i],
                        dataloaders=dataloaders["test"],
                    )
                ).numpy()
            )
        )
    target = get_all_labels(dataloaders["test"].dataset)
    return target, [Counter(x) for x in zip(*preds)]


if __name__ == "__main__":
    from hwr.network.conf import BasicTrainEvalTask, txs
    from hwr.network.CNN import CNN
    from sklearn.metrics import classification_report
    import wandb
    from pathlib import Path
    import numpy as np

    th.set_float32_matmul_precision("medium")
    # set this to true to save trained checkpoints
    # WARN: slows down the final step of wandb sync, so use false in debugging/test
    # scenarios

    # LOG_MODEL = False
    LOG_MODEL = False
    WANDB_PROJECT = "HWR_UQ"
    NUM_ESTIMATORS = 2

    run = wandb.init(project=WANDB_PROJECT)

    train_task_config = BasicTrainEvalTask(
        from_dir=Path("data/dss/monkbrill-prep-tri"),
        model_cls=CNN,
        num_estimators=NUM_ESTIMATORS,
        tx=txs,
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
                monitor="val/loss",
                patience=5,
            )
        ],
    )

    y_true, y_preds = train_and_eval_ensemble(train_task_config)
    y_preds = np.array(y_preds)
    y_pred_best = np.array([x.most_common(1)[0][0] for x in y_preds])
    y_pred_conf = np.array(
        [x.most_common(1)[0][1] / NUM_ESTIMATORS for x in y_preds]
    )
    print(y_pred_conf)
    print(y_preds)

    print(classification_report(y_true, y_pred_best))
