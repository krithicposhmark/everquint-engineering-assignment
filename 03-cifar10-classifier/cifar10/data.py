from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, eval_transform


def create_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    validation_fraction: float = 0.1,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    train_transform, eval_transform = _transforms()
    root = Path(data_dir)
    train_source = datasets.CIFAR10(
        root, train=True, download=True, transform=train_transform
    )
    validation_source = datasets.CIFAR10(
        root, train=True, download=False, transform=eval_transform
    )
    test_source = datasets.CIFAR10(
        root, train=False, download=True, transform=eval_transform
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_source), generator=generator).tolist()
    if train_limit is not None:
        if train_limit < 2:
            raise ValueError("train_limit must be at least 2")
        indices = indices[: min(train_limit, len(indices))]

    validation_size = max(1, round(len(indices) * validation_fraction))
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    test_indices = list(range(len(test_source)))
    if test_limit is not None:
        test_indices = test_indices[: min(test_limit, len(test_indices))]

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _seed_worker,
    }

    train_loader = DataLoader(
        Subset(train_source, train_indices),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_options,
    )
    validation_loader = DataLoader(
        Subset(validation_source, validation_indices), shuffle=False, **loader_options
    )
    test_loader = DataLoader(
        Subset(test_source, test_indices), shuffle=False, **loader_options
    )
    return train_loader, validation_loader, test_loader
