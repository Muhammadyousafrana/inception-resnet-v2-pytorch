"""
config.py — Central configuration for Inception-ResNet-v2 training.
All hyperparameters, paths, and toggles live here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_root: str = "./data"
    train_split: float = 0.8
    num_classes: int = 257          # Caltech-256 has 257 categories (256 + clutter)
    image_size: int = 299
    resize_size: int = 300
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    # ── DataLoader ────────────────────────────────────────────────────────────
    batch_size: int = 64
    num_workers: int = -1           # -1 → use all available CPU cores
    pin_memory: bool = True
    prefetch_factor: int = 2

    # ── Model ─────────────────────────────────────────────────────────────────
    in_channels: int = 3
    dropout: float = 0.8

    # ── Compilation ───────────────────────────────────────────────────────────
    use_compile: bool = True
    compile_mode: str = "max-autotune"  # "default" | "reduce-overhead" | "max-autotune"

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 100
    learning_rate: float = 1e-3

    # ── LR Scheduler (CosineAnnealingLR) ─────────────────────────────────────
    cosine_t_max: int = 10

    # ── Early Stopping ────────────────────────────────────────────────────────
    es_patience: int = 10
    es_delta: float = 0.0
    es_verbose: bool = True

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "best_model.pt"
    export_path: str = "inception_resnet_v2.pt2"
    runs_dir: str = "runs/inception_resnet_v2"

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    log_interval: int = 10          # Print loss every N batches

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def checkpoint_path(self) -> str:
        return str(Path(self.checkpoint_dir) / self.checkpoint_name)

    @property
    def num_workers_resolved(self) -> int:
        return os.cpu_count() if self.num_workers == -1 else self.num_workers

    def ensure_dirs(self) -> None:
        """Create checkpoint and runs directories if missing."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.runs_dir).mkdir(parents=True, exist_ok=True)
