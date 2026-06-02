import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

mnist_default_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Pad(2, padding_mode="constant"),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

mnist_none_transform = transforms.ToTensor()

cifar10_default_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ]
)

cifar10_none_transform = transforms.ToTensor()


TRANSFORMS = {
    "mnist": {"default": mnist_default_transform, "none": mnist_none_transform},
    "cifar10": {"default": cifar10_default_transform, "none": cifar10_none_transform},
}


def build_transform(name: str, dataset: str = "mnist"):
    if dataset not in TRANSFORMS:
        raise ValueError(f"Unknown dataset: {dataset}")
    transform = TRANSFORMS[dataset].get(name)
    if transform is None:
        raise ValueError(f"Unknown transform preset '{name}' for dataset '{dataset}'")
    return transform


def create_mnist_train_val_loaders(
    batch_size: int = 64,
    data_path: str = "/home/luke-padmore/Source/flow-matching-mnist/data",
    transform: str = "default",
    num_workers=4,
    shuffle=True,
) -> tuple[DataLoader, DataLoader]:
    transform = build_transform(transform, dataset="mnist")
    trainset = torchvision.datasets.MNIST(
        root=data_path, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_set = torchvision.datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return train_loader, val_loader


def create_cifar10_train_val_loaders(
    batch_size: int = 64,
    data_path: str = "/home/luke-padmore/Source/flow-matching-mnist/data",
    transform: str = "default",
    num_workers=4,
    shuffle=True,
) -> tuple[DataLoader, DataLoader]:
    transform = build_transform(transform, dataset="cifar10")
    trainset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_set = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return train_loader, val_loader
