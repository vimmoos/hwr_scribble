from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import shutil
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


class CombinedImageFolder(ImageFolder):
    def __init__(self, root, datasets):
        self.datasets = datasets
        self.classes = datasets[0].classes
        self.class_to_idx = datasets[0].class_to_idx
        self.samples = sum([dataset.samples for dataset in datasets], [])

        super(CombinedImageFolder, self).__init__(
            root, transform=None, target_transform=None
        )

    def __getitem__(self, index):
        dataset_idx, sample_idx = self.get_dataset_index(index)
        sample, target = self.datasets[dataset_idx][sample_idx]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def get_dataset_index(self, index):
        for dataset_idx, dataset in enumerate(self.datasets):
            if index < len(dataset):
                return dataset_idx, index
            else:
                index -= len(dataset)
        return (None, None)


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
        # raise StopIteration

    def __len__(self):
        return sum(self.lengths) - 1


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
