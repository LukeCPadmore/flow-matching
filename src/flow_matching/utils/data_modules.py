from __future__ import annotations

from pathlib import Path

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


_RGB_TO_XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=torch.float32,
)
_XYZ_TO_RGB = torch.tensor(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=torch.float32,
)
_XN = 0.95047
_YN = 1.0
_ZN = 1.08883
_DELTA = 6.0 / 29.0
_DELTA_CUBED = _DELTA**3
_DELTA_SQUARED = _DELTA**2


def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    return torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).clamp_min(0.0) ** 2.4,
    )


def _linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    return torch.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * rgb.clamp_min(0.0) ** (1.0 / 2.4) - 0.055,
    )


def _f_lab(t: torch.Tensor) -> torch.Tensor:
    return torch.where(
        t > _DELTA_CUBED,
        t.clamp_min(0.0) ** (1.0 / 3.0),
        t / (3.0 * _DELTA_SQUARED) + 4.0 / 29.0,
    )


def _f_lab_inv(t: torch.Tensor) -> torch.Tensor:
    return torch.where(
        t > _DELTA,
        t**3,
        3.0 * _DELTA_SQUARED * (t - 4.0 / 29.0),
    )


class RGBToLAB:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: [3, H, W] in [0, 1]
        single = img.ndim == 3
        if single:
            img = img.unsqueeze(0)

        rgb = img.clamp(0.0, 1.0).permute(0, 2, 3, 1)
        rgb_lin = _srgb_to_linear(rgb)
        xyz = rgb_lin @ _RGB_TO_XYZ.to(device=rgb.device, dtype=rgb.dtype).T

        x = xyz[..., 0] / _XN
        y = xyz[..., 1] / _YN
        z = xyz[..., 2] / _ZN

        fx = _f_lab(x)
        fy = _f_lab(y)
        fz = _f_lab(z)

        lab = torch.stack(
            [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)],
            dim=1,
        )

        lab[:, :1] /= 100.0
        lab[:, 1:] /= 128.0

        return lab.squeeze(0) if single else lab


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    # lab: [B, 3, H, W] or [3, H, W]
    single = lab.ndim == 3
    if single:
        lab = lab.unsqueeze(0)

    lab = lab.clone()
    lab[:, :1] *= 100.0
    lab[:, 1:] *= 128.0

    l = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]

    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    x = _XN * _f_lab_inv(fx)
    y = _YN * _f_lab_inv(fy)
    z = _ZN * _f_lab_inv(fz)

    xyz = torch.stack([x, y, z], dim=-1)
    rgb_lin = xyz @ _XYZ_TO_RGB.to(device=lab.device, dtype=lab.dtype).T
    rgb = _linear_to_srgb(rgb_lin).permute(0, 3, 1, 2).clamp(0.0, 1.0)
    return rgb.squeeze(0) if single else rgb


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
