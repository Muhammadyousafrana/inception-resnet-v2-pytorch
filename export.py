"""
export.py — Export a trained model with torch.export and reload for inference.

Usage (from the project root):
    python export.py                       # export using Config defaults
    python export.py --checkpoint path/to/model.pt --output my_model.pt2

Requires PyTorch ≥ 2.1 for torch.export / torch.export.save.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import Config
from model  import InceptionResNetV2


# ── Export ────────────────────────────────────────────────────────────────────

def export_model(
    checkpoint_path: str,
    export_path: str,
    config: Config,
    device: torch.device | None = None,
) -> None:
    """
    Load weights from a checkpoint, trace the model with torch.export,
    and save the ExportedProgram to disk.

    Args:
        checkpoint_path: Path to a state_dict saved by EarlyStopping.
        export_path:     Destination for the .pt2 file.
        config:          Config instance for model construction.
        device:          Export device (defaults to CUDA if available, else CPU).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[export] Loading weights from {checkpoint_path}")
    model = InceptionResNetV2(
        in_channels = config.in_channels,
        num_classes = config.num_classes,
        dropout     = config.dropout,
    )
    state = torch.load(checkpoint_path, map_location=device)
    # Handle state dicts saved from torch.compile (keys may have _orig_mod prefix)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    print(f"[export] Tracing model on device={device} …")
    dummy_input = torch.randn(1, config.in_channels, config.image_size, config.image_size, device=device)

    with torch.no_grad():
        exported = torch.export.export(model, (dummy_input,))

    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, export_path)
    print(f"[export] Saved ExportedProgram → {export_path}")


# ── Load + Inference ──────────────────────────────────────────────────────────

def load_model(
    export_path: str,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Load an ExportedProgram from disk and return a runnable module.

    Args:
        export_path: Path to the .pt2 file.
        device:      Device to move the loaded model to.

    Returns:
        Callable module ready for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[export] Loading ExportedProgram from {export_path}")
    ep     = torch.export.load(export_path)
    module = ep.module()
    module.to(device).eval()
    print(f"[export] Model loaded on {device}")
    return module


def run_inference(
    export_path: str,
    config: Config,
    device: torch.device | None = None,
) -> None:
    """Quick smoke-test: forward a random input through the loaded model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = load_model(export_path, device)

    with torch.no_grad():
        x   = torch.randn(1, config.in_channels, config.image_size, config.image_size, device=device)
        out = module(x)

    print(f"[export] Inference OK — output shape: {out.shape}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export or test Inception-ResNet-v2")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to state_dict .pt file (default: config.checkpoint_path)")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output path for .pt2 file (default: config.export_path)")
    parser.add_argument("--test",       action="store_true",
                        help="After export, run a smoke-test inference")
    args = parser.parse_args()

    cfg = Config()
    ckpt = args.checkpoint or cfg.checkpoint_path
    out  = args.output     or cfg.export_path

    export_model(ckpt, out, cfg)

    if args.test:
        run_inference(out, cfg)
