"""Simple dataset loader that works with folder-based image datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

NORMALIZATION_PRESETS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "dinov2": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    "tiny-imagenet": ((0.480, 0.448, 0.398), (0.277, 0.269, 0.282)),
}


@dataclass
class DatasetSpec:
    name: str
    root: str = "/speedy/datasets"
    train_split: str = "train"
    val_split: Optional[str] = None
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    augment: bool = True
    normalization: str | Tuple[Iterable[float], Iterable[float]] = "imagenet"
    drop_last: bool = True


@dataclass
class DatasetInfo:
    name: str
    train_samples: int
    val_samples: int
    num_classes: int
    classes: Iterable[str]
    train_split: str
    val_split: str
    root: Path


def _resolve_normalization(spec: DatasetSpec) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    norm = spec.normalization
    if isinstance(norm, str):
        key = norm.lower()
        if key not in NORMALIZATION_PRESETS:
            raise KeyError(f"Unknown normalization preset: {norm}")
        return NORMALIZATION_PRESETS[key]
    mean, std = norm
    return tuple(mean), tuple(std)  # type: ignore[arg-type]


def _transforms(spec: DatasetSpec):
    mean, std = _resolve_normalization(spec)

    if spec.augment:
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(spec.image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize(int(spec.image_size * 1.1)),
                transforms.CenterCrop(spec.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(spec.image_size * 1.1)),
            transforms.CenterCrop(spec.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_tf, eval_tf


def _resolve_split(dataset_dir: Path, preferred: Optional[str], fallbacks: Tuple[str, ...]) -> str:
    if preferred:
        path = dataset_dir / preferred
        if path.is_dir():
            return preferred
    for candidate in fallbacks:
        path = dataset_dir / candidate
        if path.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find a valid split under {dataset_dir}")


def create_dataloaders(spec: DatasetSpec) -> Tuple[DataLoader, DataLoader, DatasetInfo]:
    dataset_dir = Path(spec.root) / spec.name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    train_split = _resolve_split(dataset_dir, spec.train_split, ("train", "training"))
    val_split = _resolve_split(dataset_dir, spec.val_split, ("val", "validation", "test"))

    train_tf, val_tf = _transforms(spec)

    train_ds = ImageFolder(dataset_dir / train_split, transform=train_tf)
    val_ds = ImageFolder(dataset_dir / val_split, transform=val_tf)

    if train_ds.classes != val_ds.classes:
        raise RuntimeError(f"Class mismatch between splits for dataset {spec.name}")

    persistent = spec.persistent_workers and spec.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=spec.num_workers,
        pin_memory=spec.pin_memory,
        drop_last=spec.drop_last,
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=max(spec.batch_size * 2, 1),
        shuffle=False,
        num_workers=spec.num_workers,
        pin_memory=spec.pin_memory,
        drop_last=False,
        persistent_workers=persistent,
    )

    info = DatasetInfo(
        name=spec.name,
        train_samples=len(train_ds.samples),
        val_samples=len(val_ds.samples),
        num_classes=len(train_ds.classes),
        classes=train_ds.classes,
        train_split=train_split,
        val_split=val_split,
        root=dataset_dir,
    )

    return train_loader, val_loader, info


__all__ = [
    "DatasetSpec",
    "DatasetInfo",
    "NORMALIZATION_PRESETS",
    "create_dataloaders",
]
