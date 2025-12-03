"""Dataset helpers for mouth-operator classification."""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Union

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

__all__ = ["build_dataloaders", "get_transforms"]


def get_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train/val transform pipelines."""
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, val_tfms


def build_dataloaders(
    data_root: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """Create train/val dataloaders from an ImageFolder-style directory.

    Expected layout:
      data_root/
        train/{class}/image.png
        val/{class}/image.png
    """
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected train/ and val/ under {data_root}, found {list(data_root.iterdir()) if data_root.exists() else 'missing'}")

    train_tfms, val_tfms = get_transforms(image_size=image_size)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    return train_loader, val_loader, idx_to_class
