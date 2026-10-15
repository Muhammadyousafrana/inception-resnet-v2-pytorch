"""
utils/plot.py — Visualization utilities: sample inspection and training curves.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision


# ── Normalisation helpers ─────────────────────────────────────────────────────

def denormalize(
    tensor: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std:  Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Reverse ImageNet normalisation on a (C, H, W) float tensor.

    Returns a *cloned* tensor so the original is not mutated.
    Values are clamped to [0, 1] to avoid matplotlib clipping warnings.
    """
    t = tensor.clone().float()
    for channel, (m, s) in enumerate(zip(mean, std)):
        t[channel] = t[channel] * s + m
    return torch.clamp(t, 0.0, 1.0)


# ── Sample grid ───────────────────────────────────────────────────────────────

def visualize_samples(
    dataset: torch.utils.data.Dataset,
    categories: List[str],
    num_samples: int = 8,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std:  Sequence[float] = (0.229, 0.224, 0.225),
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 4),
) -> None:
    """
    Display a row of sample images with their class labels.

    Args:
        dataset:     Any Dataset whose items are (tensor, label_int).
        categories:  List mapping label int → class name string.
        num_samples: How many images to show.
        mean / std:  ImageNet stats used to de-normalise before display.
        save_path:   If given, saves the figure to disk instead of showing.
        figsize:     Figure dimensions (width, height) in inches.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    for i in range(num_samples):
        img, label = dataset[i]
        img_np = denormalize(img, mean, std).permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(categories[label], fontsize=8)
        axes[i].axis("off")

    plt.suptitle("Sample images from dataset", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[plot] Saved sample grid → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: List[float],
    val_losses:   List[float],
    train_accs:   Optional[List[float]] = None,
    val_accs:     Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot loss (and optionally accuracy) curves over epochs.

    Args:
        train_losses: Per-epoch average training loss.
        val_losses:   Per-epoch average validation loss.
        train_accs:   Per-epoch training accuracy (optional).
        val_accs:     Per-epoch validation accuracy (optional).
        save_path:    If given, saves the figure instead of showing it.
    """
    has_acc = train_accs is not None and val_accs is not None
    n_plots = 2 if has_acc else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss
    axes[0].plot(epochs, train_losses, label="Train loss", marker="o", markersize=3)
    axes[0].plot(epochs, val_losses,   label="Val loss",   marker="o", markersize=3)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy (optional)
    if has_acc:
        axes[1].plot(epochs, train_accs, label="Train acc", marker="o", markersize=3)
        axes[1].plot(epochs, val_accs,   label="Val acc",   marker="o", markersize=3)
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Top-1 accuracy (%)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[plot] Saved training curves → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Batch preview from DataLoader ─────────────────────────────────────────────

def visualize_batch(
    loader: torch.utils.data.DataLoader,
    categories: List[str],
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std:  Sequence[float] = (0.229, 0.224, 0.225),
    max_images: int = 16,
    save_path: Optional[str] = None,
) -> None:
    """
    Show a grid of images taken from the first batch of a DataLoader.
    Useful for sanity-checking the full augmentation + normalisation pipeline.
    """
    images, labels = next(iter(loader))
    images = images[:max_images]
    labels = labels[:max_images]

    # Denormalise each image
    grid_imgs = torch.stack([denormalize(img, mean, std) for img in images])
    grid = torchvision.utils.make_grid(grid_imgs, nrow=8, padding=2)

    fig, ax = plt.subplots(figsize=(16, max_images // 8 * 2 + 2))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    title = "  ".join(categories[l.item()] for l in labels)
    ax.set_title(title, fontsize=7, wrap=True)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[plot] Saved batch grid → {save_path}")
    else:
        plt.show()
    plt.close(fig)
