from __future__ import annotations

from pathlib import Path

import kornia
import lightning.pytorch as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class DropLabelDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, _ = self.dataset[idx]
        return x


class RGBToLAB:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.cpu()
        single = img.ndim == 3

        if single:
            img = img.unsqueeze(0)

        lab = kornia.color.rgb_to_lab(img)

        lab[:, :1] = lab[:, :1] / 100.0
        lab[:, 1:] = lab[:, 1:] / 128.0

        return lab.squeeze(0) if single else lab


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    lab = lab.clone()
    lab[:, :1] = lab[:, :1] * 100.0
    lab[:, 1:] = lab[:, 1:] * 128.0

    return kornia.color.lab_to_rgb(lab).clamp(0.0, 1.0)


MNIST_DEFAULT_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Pad(2, padding_mode="constant"),
        v2.Normalize((0.5,), (0.5,)),
    ]
)

MNIST_NONE_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

CIFAR10_DEFAULT_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)

CIFAR10_NONE_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

CIFAR10_LAB_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        RGBToLAB(),
    ]
)


def _make_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _mnist_transform(name: str):
    if name == "default":
        return MNIST_DEFAULT_TRANSFORM
    if name == "none":
        return MNIST_NONE_TRANSFORM
    raise ValueError(f"Unknown MNIST transform preset: {name}")


def _cifar10_transform(name: str):
    if name == "default":
        return CIFAR10_DEFAULT_TRANSFORM
    if name == "LAB":
        return CIFAR10_LAB_TRANSFORM
    if name == "none":
        return CIFAR10_NONE_TRANSFORM
    raise ValueError(f"Unknown CIFAR10 transform preset: {name}")


class MNISTDataModule(pl.LightningDataModule):
    dataset_name = "mnist"
    image_shape = (1, 32, 32)
    num_classes = 10

    def __init__(
        self,
        batch_size: int = 64,
        data_path: str | Path = "data",
        num_workers: int = 0,
        transform: str = "default",
        shuffle: bool = True,
        drop_labels: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None
        self.sample_mean = None
        self.sample_std = None

    def prepare_data(self) -> None:
        torchvision.datasets.MNIST(self.hparams.data_path, train=True, download=True)
        torchvision.datasets.MNIST(self.hparams.data_path, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        transform = _mnist_transform(self.hparams.transform)

        trainset = torchvision.datasets.MNIST(
            root=self.hparams.data_path,
            train=True,
            download=False,
            transform=transform,
        )

        valset = torchvision.datasets.MNIST(
            root=self.hparams.data_path,
            train=False,
            download=False,
            transform=transform,
        )

        if self.hparams.drop_labels:
            trainset = DropLabelDataset(trainset)
            valset = DropLabelDataset(valset)

        self.train_dataset = trainset
        self.val_dataset = valset

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup("fit")

        return _make_loader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup("fit")

        return _make_loader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


class CIFAR10DataModule(pl.LightningDataModule):
    dataset_name = "cifar10"
    image_shape = (3, 32, 32)
    num_classes = 10

    def __init__(
        self,
        batch_size: int = 64,
        data_path: str | Path = "data",
        num_workers: int = 0,
        transform: str = "default",
        shuffle: bool = True,
        drop_labels: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None
        self.sample_mean, self.sample_std = self._sample_stats(transform)

    @staticmethod
    def _sample_stats(
        transform: str,
    ) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
        if transform == "default":
            return CIFAR10_MEAN, CIFAR10_STD
        return None, None

    def prepare_data(self) -> None:
        torchvision.datasets.CIFAR10(self.hparams.data_path, train=True, download=True)
        torchvision.datasets.CIFAR10(self.hparams.data_path, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        transform = _cifar10_transform(self.hparams.transform)

        trainset = torchvision.datasets.CIFAR10(
            root=self.hparams.data_path,
            train=True,
            download=False,
            transform=transform,
        )

        valset = torchvision.datasets.CIFAR10(
            root=self.hparams.data_path,
            train=False,
            download=False,
            transform=transform,
        )

        if self.hparams.drop_labels:
            trainset = DropLabelDataset(trainset)
            valset = DropLabelDataset(valset)

        self.train_dataset = trainset
        self.val_dataset = valset

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup("fit")

        return _make_loader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup("fit")

        return _make_loader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


class CIFAR10ColouriserDataModule(CIFAR10DataModule):
    dataset_name = "cifar10LAB"
    image_shape = (3, 32, 32)

    def __init__(
        self,
        batch_size: int = 64,
        data_path: str | Path = "data",
        num_workers: int = 0,
        transform: str = "LAB",
        shuffle: bool = True,
        drop_labels: bool = True,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            data_path=data_path,
            num_workers=num_workers,
            transform=transform,
            shuffle=shuffle,
            drop_labels=drop_labels,
        )
