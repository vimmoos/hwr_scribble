from dataclasses import dataclass
from typing import Callable, List, Optional
from pathlib import Path
import lightning.pytorch as pl
from torchvision import transforms
from hwr.network.CNN import CNN


@dataclass
class BasicTrainEvalTask:
    # input
    from_dir: Path
    tx: Callable
    # model
    model_cls: pl.LightningModule = CNN
    num_estimators: int = 5
    # data
    num_classes: int = 27
    test_size: float = 0.3
    val_size_of_train: float = 0.2
    random_state: Optional[int] = 42
    stratify: bool = True
    shuffle: bool = True
    batch_size: int = 64
    # tracking
    wandb_project: str = "HWR"
    log_model: bool = True
    # training
    max_epochs: int = 2000
    callbacks: Optional[List[pl.Callback]] = None
    return_test_set_predictions = True


txs = transforms.Compose(
    [
        transforms.Resize(
            (28, 28),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)
