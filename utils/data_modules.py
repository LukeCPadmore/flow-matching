from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

MNIST_DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Pad(2, padding_mode="constant"),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
MNIST_NONE_TRANSFORM = transforms.ToTensor()

CIFAR10_DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)
CIFAR10_NONE_TRANSFORM = transforms.ToTensor()


def _make_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
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
    if name == "none":
        return CIFAR10_NONE_TRANSFORM
    raise ValueError(f"Unknown CIFAR10 transform preset: {name}")


class MNISTDataModule(pl.LightningDataModule):
    dataset_name = "mnist"
    image_shape = (1, 32, 32)
    num_classes = 10
    sample_mean = None
    sample_std = None

    def __init__(
        self,
        batch_size: int = 64,
        data_path: str | Path = "data",
        num_workers: int = 0,
        transform: str = "default",
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None

    def _make_datasets(self):
        transform = _mnist_transform(self.hparams.transform)
        trainset = torchvision.datasets.MNIST(
            root=self.hparams.data_path,
            train=True,
            download=True,
            transform=transform,
        )
        val_set = torchvision.datasets.MNIST(
            root=self.hparams.data_path,
            train=False,
            download=True,
            transform=transform,
        )
        return trainset, val_set

    def prepare_data(self) -> None:
        self._make_datasets()

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is None or self.val_dataset is None:
            self.train_dataset, self.val_dataset = self._make_datasets()

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
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None
        self.sample_mean, self.sample_std = self._sample_stats(transform)

    @staticmethod
    def _sample_stats(transform: str) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
        if transform == "default":
            return CIFAR10_MEAN, CIFAR10_STD
        return None, None

    def _make_datasets(self):
        transform = _cifar10_transform(self.hparams.transform)
        trainset = torchvision.datasets.CIFAR10(
            root=self.hparams.data_path,
            train=True,
            download=True,
            transform=transform,
        )
        val_set = torchvision.datasets.CIFAR10(
            root=self.hparams.data_path,
            train=False,
            download=True,
            transform=transform,
        )
        return trainset, val_set

    def prepare_data(self) -> None:
        self._make_datasets()

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is None or self.val_dataset is None:
            self.train_dataset, self.val_dataset = self._make_datasets()

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
