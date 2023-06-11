from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import shutil
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = [0] + list(
            torch.cumsum(torch.tensor(self.lengths), dim=0)
        )

    def __getitem__(self, index):
        for dataset_idx, cumulative_len in enumerate(
            self.cumulative_lengths[:-1]
        ):
            if (
                index >= cumulative_len
                and index < self.cumulative_lengths[dataset_idx + 1]
            ):
                return self.datasets[dataset_idx][index - cumulative_len]
        raise IndexError("Index out of range")

    def __len__(self):
        return sum(self.lengths)


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
    for i, (X, y) in enumerate(tqdm(dataset, total=len(dataset))):
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
