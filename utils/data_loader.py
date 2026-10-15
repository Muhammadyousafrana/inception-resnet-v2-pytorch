"""
utils/data_loader.py — Transforms, dataset wrappers, and DataLoader factory.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(
    is_train: bool,
    resize: int = 300,
    crop: int = 299,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...]  = (0.229, 0.224, 0.225),
) -> v2.Compose:
    """
    Returns a torchvision v2 transform pipeline.

    Training adds RandomHorizontalFlip for augmentation.
    All transforms run on CPU inside DataLoader workers.
    """
    base = [
        v2.Resize(resize),
        v2.CenterCrop(crop),
    ]
    if is_train:
        base.append(v2.RandomHorizontalFlip(p=0.5))
    base += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ]
    return v2.Compose(base)


# ── Dataset Wrapper ───────────────────────────────────────────────────────────

class SubsetWithTransform(Dataset):
    """
    Wraps a Subset returned by random_split and applies a per-split transform.

    This is necessary because torchvision datasets accept a single global
    transform at construction time, but we need different augmentations for
    train vs. val subsets derived from the same dataset object.
    """

    def __init__(self, subset: torch.utils.data.Subset, transform: v2.Compose) -> None:
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x, y = self.subset[idx]
        return self.transform(x), y


# ── DataLoader Factory ────────────────────────────────────────────────────────

def get_dataloaders(
    data_root: str = "./data",
    train_split: float = 0.8,
    batch_size: int = 64,
    num_workers: int = -1,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    resize: int = 300,
    crop: int = 299,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...]  = (0.229, 0.224, 0.225),
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, torchvision.datasets.Caltech256]:
    """
    Downloads (if needed) and splits Caltech-256, returns:
        (train_loader, val_loader, full_dataset)

    The full dataset is returned so callers can access `.categories`.

    Args:
        data_root:       Directory to download/cache the dataset.
        train_split:     Fraction of data used for training.
        batch_size:      Samples per batch.
        num_workers:     DataLoader workers; -1 uses all CPU cores.
        pin_memory:      Use pinned memory for faster GPU transfer.
        prefetch_factor: Batches to prefetch per worker.
        resize:          Resize shorter side to this before CenterCrop.
        crop:            Final square crop size (must match model input).
        mean / std:      ImageNet normalisation statistics.
        seed:            Seed for reproducible train/val split.

    Returns:
        train_loader, val_loader, full_dataset
    """
    if num_workers == -1:
        num_workers = os.cpu_count()

    # ── Download + split ──────────────────────────────────────────────────────
    full_ds = torchvision.datasets.Caltech256(root=data_root, download=True)

    n_train = int(train_split * len(full_ds))
    n_val   = len(full_ds) - n_train

    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(full_ds, [n_train, n_val], generator=generator)

    # ── Wrap with per-split transforms ────────────────────────────────────────
    train_tf = get_transforms(is_train=True,  resize=resize, crop=crop, mean=mean, std=std)
    val_tf   = get_transforms(is_train=False, resize=resize, crop=crop, mean=mean, std=std)

    train_dataset = SubsetWithTransform(train_sub, train_tf)
    val_dataset   = SubsetWithTransform(val_sub,   val_tf)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loader_kwargs = dict(
        batch_size      = batch_size,
        num_workers     = num_workers,
        pin_memory      = pin_memory,
        prefetch_factor = prefetch_factor if num_workers > 0 else None,
        persistent_workers = num_workers > 0,
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    print(
        f"[data] {len(train_dataset):,} train / {len(val_dataset):,} val  |  "
        f"batch={batch_size}  workers={num_workers}"
    )
    return train_loader, val_loader, full_ds
