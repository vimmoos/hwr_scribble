from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import shutil
import torch
import numpy as np


def format_with_leading_zeros(x, n_fixed_chars):
    fmt = "{:0" + str(n_fixed_chars) + "}"
    return fmt.format(x)


def dataset_dump(
    dataset, dest_dir, class_to_name_mapping, img_ext=".pgm", rm_dest=True
):
    root = Path(dest_dir)
    if rm_dest and root.exists():
        print("removing old contents")
        shutil.rmtree(root)

    img_ext = img_ext if img_ext.startswith(".") else "." + img_ext
    _n_digits = 1 + len(str(len(dataset)))
    for i, (X, y) in enumerate(tqdm(dataset)):
        class_name = class_to_name_mapping[y]
        class_dir = root / class_name
        class_dir.mkdir(exist_ok=True, parents=True)
        file_dest = (
            class_dir / f"{format_with_leading_zeros(i, _n_digits)}{img_ext}"
        )
        img = transforms.ToPILImage()(X)
        img.save(file_dest)


def to_cv(t: torch.Tensor):
    return (t.numpy() * 255).astype("uint8")


def cv_to_tensor(x: np.array):
    return torch.as_tensor(x / 255)
