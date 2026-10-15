"""
train.py — Main training entry point.

Usage:
    python train.py                    # default Config
    python train.py --epochs 50 --lr 3e-4 --batch-size 32

The Trainer class encapsulates one complete experiment:
    • model build + optional torch.compile
    • optimizer + LR scheduler
    • per-epoch train / val loops with tqdm
    • TensorBoard logging
    • EarlyStopping + best-model checkpointing
    • final training-curve plot saved to disk
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from model  import InceptionResNetV2
from utils  import (
    EarlyStopping,
    TensorBoardLogger,
    get_dataloaders,
    plot_training_curves,
    visualize_samples,
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Device detection ─────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """
    Self-contained training loop.

    Args:
        config: Frozen Config dataclass instance.
    """

    def __init__(self, config: Config) -> None:
        self.cfg    = config
        self.device = get_device()
        print(f"[trainer] Device: {self.device}")

        set_seed(config.seed)
        config.ensure_dirs()

        # ── Data ──────────────────────────────────────────────────────────────
        self.train_loader, self.val_loader, self.full_ds = get_dataloaders(
            data_root      = config.data_root,
            train_split    = config.train_split,
            batch_size     = config.batch_size,
            num_workers    = config.num_workers_resolved,
            pin_memory     = config.pin_memory,
            prefetch_factor= config.prefetch_factor,
            resize         = config.resize_size,
            crop           = config.image_size,
            mean           = config.mean,
            std            = config.std,
            seed           = config.seed,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = InceptionResNetV2(
            in_channels = config.in_channels,
            num_classes = config.num_classes,
            dropout     = config.dropout,
        ).to(self.device)

        print(f"[trainer] Parameters: {self.model.count_parameters():,}")

        if config.use_compile:
            print(f"[trainer] Compiling model (mode={config.compile_mode}) …")
            self.model = torch.compile(self.model, mode=config.compile_mode)

        # ── Optimiser + Scheduler ─────────────────────────────────────────────
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.cosine_t_max
        )
        self.loss_fn = nn.CrossEntropyLoss()

        # ── Callbacks ─────────────────────────────────────────────────────────
        self.early_stopping = EarlyStopping(
            patience = config.es_patience,
            delta    = config.es_delta,
            path     = config.checkpoint_path,
            verbose  = config.es_verbose,
        )
        self.logger = TensorBoardLogger(config.runs_dir)

        # ── History ───────────────────────────────────────────────────────────
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── Single epoch helpers ──────────────────────────────────────────────────

    def _train_epoch(self) -> Tuple[float, float]:
        """Run one training epoch. Returns (avg_loss, accuracy %)."""
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for images, labels in tqdm(self.train_loader, desc="  train", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss   = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        return total_loss / len(self.train_loader), 100.0 * correct / total

    @torch.no_grad()
    def _val_epoch(self) -> Tuple[float, float]:
        """Run one validation epoch. Returns (avg_loss, accuracy %)."""
        self.model.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for images, labels in tqdm(self.val_loader, desc="  val  ", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss   = self.loss_fn(logits, labels)

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        return total_loss / len(self.val_loader), 100.0 * correct / total

    # ── Main fit loop ─────────────────────────────────────────────────────────

    def fit(self) -> None:
        print(f"\n[trainer] Starting training for {self.cfg.epochs} epochs …\n")
        t0 = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            t_ep = time.time()

            train_loss, train_acc = self._train_epoch()
            val_loss,   val_acc   = self._val_epoch()

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # ── Logging ───────────────────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            self.logger.log_epoch(
                epoch,
                train_loss = train_loss,
                val_loss   = val_loss,
                train_acc  = train_acc,
                val_acc    = val_acc,
                lr         = current_lr,
            )

            elapsed = time.time() - t_ep
            print(
                f"Epoch {epoch:3d}/{self.cfg.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%  "
                f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
            )

            # ── Early stopping ────────────────────────────────────────────────
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"\n[trainer] Early stopping triggered at epoch {epoch}.")
                break

        total_time = time.time() - t0
        print(f"\n[trainer] Training complete in {total_time/60:.1f} min.")

        self.logger.close()

        # ── Post-training ─────────────────────────────────────────────────────
        plot_training_curves(
            self.history["train_loss"],
            self.history["val_loss"],
            self.history["train_acc"],
            self.history["val_acc"],
            save_path="runs/training_curves.png",
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Inception-ResNet-v2 on Caltech-256")
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None, dest="learning_rate")
    p.add_argument("--batch-size",  type=int,   default=None, dest="batch_size")
    p.add_argument("--data-root",   type=str,   default=None, dest="data_root")
    p.add_argument("--no-compile",  action="store_true")
    p.add_argument("--seed",        type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = Config()

    # CLI overrides
    if args.epochs        is not None: cfg.epochs        = args.epochs
    if args.learning_rate is not None: cfg.learning_rate = args.learning_rate
    if args.batch_size    is not None: cfg.batch_size    = args.batch_size
    if args.data_root     is not None: cfg.data_root     = args.data_root
    if args.no_compile:                cfg.use_compile   = False
    if args.seed          is not None: cfg.seed          = args.seed

    trainer = Trainer(cfg)

    # Visualise a few samples before training starts
    visualize_samples(
        trainer.train_loader.dataset,
        trainer.full_ds.categories,
        num_samples=8,
        save_path="runs/sample_grid.png",
    )

    trainer.fit()
