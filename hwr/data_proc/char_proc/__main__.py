from pathlib import Path

from hwr.data_proc.char_proc.utils import (
    dataset_dump,
    CombinedImageFolder,
    CombinedDataset,
)

import typer
from torchvision.datasets import ImageFolder
from typing_extensions import Annotated
from hwr.utils.resolve import VarRef
from collections.abc import Iterable


def tx_dataset(
    from_root: Annotated[
        Path,
        typer.Option(
            None,
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    to_root: Annotated[
        Path,
        typer.Option(
            None,
            dir_okay=True,
            file_okay=False,
            writable=True,
        ),
    ],
    tx_spec: str = typer.Option(None),
    img_ext: str = ".pgm",
    rm_dest: bool = True,
):
    txs = VarRef(tx_spec).resolve()
    if not isinstance(txs, Iterable):
        txs = [txs]

    datasets = [
        ImageFolder(
            from_root,
            transform=tx,
        )
        for tx in txs
    ]
    dataset = CombinedDataset(datasets)

    dataset_dump(
        dataset,
        to_root,
        class_to_name_mapping=datasets[0].classes,
        img_ext=img_ext,
        rm_dest=rm_dest,
    )
    # -- conversion utils --

    # python -m hwr.data_proc.char_proc --tx-spec       --img-ext ".pgm"  --rm-dest


tx_dataset(
    from_root="./data/original/dss/monkbrill",
    to_root="./data/dss/monkbrill-prep-final",
    rm_dest=True,
    tx_spec="hwr.data_proc.char_proc:char_transforms",
)
if __name__ == "__main__":
    typer.run(tx_dataset)
