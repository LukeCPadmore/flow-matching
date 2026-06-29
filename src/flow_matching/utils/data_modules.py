from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

from .color import RGBToLAB, lab_to_rgb_torch as lab_to_rgb
from .dataset_weights import compute_image_rarity, compute_inv_pixel_weighting


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

OXFORD_PETS_DEFAULT_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((128, 128)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

OXFORD_PETS_NONE_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((128, 128)),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

OXFORD_PETS_LAB_TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((128, 128)),
        v2.ToDtype(torch.float32, scale=True),
        RGBToLAB(),
    ]
)


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler=None,
):
    pin_memory = torch.cuda.is_available() and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
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


def _oxford_pets_transform(name: str):
    if name == "default":
        return OXFORD_PETS_DEFAULT_TRANSFORM
    if name == "LAB":
        return OXFORD_PETS_LAB_TRANSFORM
    if name == "none":
        return OXFORD_PETS_NONE_TRANSFORM
    raise ValueError(f"Unknown Oxford Pets transform preset: {name}")


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
        use_weighted_sampler: bool = False,
        weight_alpha: float = 0.5,
        weight_eps: float = 1e-3,
        weight_bins: int = 32,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            data_path=data_path,
            num_workers=num_workers,
            transform=transform,
            shuffle=shuffle,
            drop_labels=drop_labels,
        )
        self.use_weighted_sampler = bool(use_weighted_sampler)
        self.weight_alpha = float(weight_alpha)
        self.weight_eps = float(weight_eps)
        self.weight_bins = int(weight_bins)
        self.train_sampler = None
        self.train_sample_weights = None

    def _build_weighted_sampler(self, trainset) -> WeightedRandomSampler:
        # Use a single-process loader for sampler precompute to avoid
        # multiprocessing/CUDA initialization issues on the current machine.
        weight_loader = _make_loader(
            trainset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
        )
        rarity, bins = compute_inv_pixel_weighting(
            weight_loader,
            alpha=self.weight_alpha,
            eps=self.weight_eps,
            bins=self.weight_bins,
        )
        weights = compute_image_rarity(weight_loader, rarity, bins)
        if len(weights) != len(trainset):
            raise RuntimeError("Expected one sampler weight per training image.")
        self.train_sample_weights = weights
        return WeightedRandomSampler(
            weights=weights.double(),
            num_samples=len(weights),
            replacement=True,
        )

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

        if self.use_weighted_sampler:
            self.train_sampler = self._build_weighted_sampler(trainset)

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
            sampler=self.train_sampler if self.use_weighted_sampler else None,
        )


class OxfordPetsDataModule(pl.LightningDataModule):
    dataset_name = "oxford_pets"
    image_shape = (3, 128, 128)
    num_classes = 37

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
        self.sample_mean = (0.5, 0.5, 0.5) if transform == "default" else None
        self.sample_std = (0.5, 0.5, 0.5) if transform == "default" else None

    def prepare_data(self) -> None:
        torchvision.datasets.OxfordIIITPet(
            self.hparams.data_path,
            split="trainval",
            target_types="category",
            download=True,
        )
        torchvision.datasets.OxfordIIITPet(
            self.hparams.data_path,
            split="test",
            target_types="category",
            download=True,
        )

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        transform = _oxford_pets_transform(self.hparams.transform)

        trainset = torchvision.datasets.OxfordIIITPet(
            root=self.hparams.data_path,
            split="trainval",
            target_types="category",
            download=True,
            transform=transform,
        )
        valset = torchvision.datasets.OxfordIIITPet(
            root=self.hparams.data_path,
            split="test",
            target_types="category",
            download=True,
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


class OxfordPetsColouriserDataModule(OxfordPetsDataModule):
    dataset_name = "oxford_petsLAB"
    image_shape = (3, 128, 128)

    def __init__(
        self,
        batch_size: int = 32,
        data_path: str | Path = "data",
        num_workers: int = 0,
        transform: str = "LAB",
        shuffle: bool = True,
        drop_labels: bool = True,
        use_weighted_sampler: bool = False,
        weight_alpha: float = 0.5,
        weight_eps: float = 1e-3,
        weight_bins: int = 32,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            data_path=data_path,
            num_workers=num_workers,
            transform=transform,
            shuffle=shuffle,
            drop_labels=drop_labels,
        )
        self.use_weighted_sampler = bool(use_weighted_sampler)
        self.weight_alpha = float(weight_alpha)
        self.weight_eps = float(weight_eps)
        self.weight_bins = int(weight_bins)
        self.train_sampler = None
        self.train_sample_weights = None

    def _build_weighted_sampler(self, trainset) -> WeightedRandomSampler:
        # Use a single-process loader for sampler precompute to avoid
        # multiprocessing/CUDA initialization issues on the current machine.
        weight_loader = _make_loader(
            trainset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
        )
        rarity, bins = compute_inv_pixel_weighting(
            weight_loader,
            alpha=self.weight_alpha,
            eps=self.weight_eps,
            bins=self.weight_bins,
        )
        weights = compute_image_rarity(weight_loader, rarity, bins)
        if len(weights) != len(trainset):
            raise RuntimeError(
                "Expected one sampler weight per training image."
            )
        self.train_sample_weights = weights
        return WeightedRandomSampler(
            weights=weights.double(),
            num_samples=len(weights),
            replacement=True,
        )

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        transform = _oxford_pets_transform(self.hparams.transform)

        trainset = torchvision.datasets.OxfordIIITPet(
            root=self.hparams.data_path,
            split="trainval",
            target_types="category",
            download=True,
            transform=transform,
        )
        valset = torchvision.datasets.OxfordIIITPet(
            root=self.hparams.data_path,
            split="test",
            target_types="category",
            download=True,
            transform=transform,
        )

        if self.use_weighted_sampler:
            self.train_sampler = self._build_weighted_sampler(trainset)

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
            sampler=self.train_sampler if self.use_weighted_sampler else None,
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
