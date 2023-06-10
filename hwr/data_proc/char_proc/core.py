from pathlib import Path
from hwr.data_proc.char_proc.utils import dataset_dump


import typer
from torchvision.datasets import ImageFolder
from typing_extensions import Annotated
from hwr2.common.resolve import VarRef


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
    tx = VarRef(tx_spec).resolve()
    print("tx=", tx)
    dataset = ImageFolder(from_root, transform=tx)
    dataset_dump(
        dataset,
        to_root,
        class_to_name_mapping=dataset.classes,
        img_ext=img_ext,
        rm_dest=rm_dest,
    )


# -- conversion utils --


if __name__ == "__main__":
    typer.run(tx_dataset)
