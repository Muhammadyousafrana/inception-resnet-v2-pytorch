"""
utils/tensorboard.py — TensorBoard logging wrapper and EarlyStopping callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


# ── TensorBoard logger ────────────────────────────────────────────────────────

class TensorBoardLogger:
    """
    Thin wrapper around SummaryWriter with convenience methods.

    Usage:
        logger = TensorBoardLogger("runs/my_experiment")
        logger.log_scalars({"Loss/train": 1.2, "Loss/val": 1.5}, step=epoch)
        logger.log_lr(optimizer, step=epoch)
        logger.close()

    Then launch:
        tensorboard --logdir runs
    """

    def __init__(self, log_dir: str) -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer   = SummaryWriter(log_dir=log_dir)
        self.log_dir  = log_dir
        print(f"[tensorboard] Logging to: {log_dir}")
        print(f"[tensorboard] Run:  tensorboard --logdir {Path(log_dir).parent}")

    # ── Scalars ───────────────────────────────────────────────────────────────

    def log_scalars(self, scalars: Dict[str, float], step: int) -> None:
        """Write multiple scalar tags in one call."""
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, global_step=step)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> None:
        """Convenience: log standard per-epoch metrics in one shot."""
        self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        if train_acc is not None and val_acc is not None:
            self.writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        if lr is not None:
            self.writer.add_scalar("LR", lr, epoch)

    def log_lr(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        """Log current learning rate(s) from an optimizer."""
        for i, pg in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"LR/group_{i}", pg["lr"], global_step=step)

    # ── Model graph ───────────────────────────────────────────────────────────

    def log_graph(self, model: torch.nn.Module, dummy_input: torch.Tensor) -> None:
        """Add model computation graph (viewable in the GRAPHS tab)."""
        self.writer.add_graph(model, dummy_input)

    # ── Histograms ────────────────────────────────────────────────────────────

    def log_weights(self, model: torch.nn.Module, step: int) -> None:
        """Log weight and gradient histograms for every named parameter."""
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"weights/{name}", param.data, step)
            if param.grad is not None:
                self.writer.add_histogram(f"grads/{name}", param.grad, step)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
        print("[tensorboard] Writer closed.")

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Also saves the best model checkpoint to disk.

    Args:
        patience:   Number of epochs with no improvement before stopping.
        delta:      Minimum decrease in val loss to count as improvement.
        path:       Where to save the best model state_dict.
        verbose:    Print a message when a new best is found.
    """

    def __init__(
        self,
        patience: int  = 10,
        delta:    float = 0.0,
        path:     str  = "checkpoints/best_model.pt",
        verbose:  bool = True,
    ) -> None:
        self.patience   = patience
        self.delta      = delta
        self.path       = path
        self.verbose    = verbose

        self.counter    = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = np.inf

        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"[early_stop] No improvement {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self._save(val_loss, model)
            self.counter = 0

    def _save(self, val_loss: float, model: torch.nn.Module) -> None:
        if self.verbose:
            print(
                f"[early_stop] Val loss improved "
                f"({self.val_loss_min:.6f} → {val_loss:.6f}). Saving checkpoint."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_best(self, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        """Restore the best weights into a model and return it."""
        model.load_state_dict(torch.load(self.path, map_location=device))
        print(f"[early_stop] Loaded best weights from {self.path}")
        return model
