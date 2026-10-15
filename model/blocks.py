"""
model/blocks.py — Individual building blocks for Inception-ResNet-v2.

Architecture reference:
  https://arxiv.org/abs/1602.07261 (Szegedy et al., 2017)

Channel flow:
  Input(3, 299, 299)
    → StemBlock            → (384, 34, 34)
    → 5× InceptionResNetA  → (384, 34, 34)
    → ReductionA           → (1152, 16, 16)
    → 10× InceptionResNetB → (1152, 16, 16)
    → ReductionB           → (2144, 8, 8)
    → 5× InceptionResNetC  → (2144, 8, 8)
"""

import torch
import torch.nn as nn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch: int, out_ch: int, **kwargs) -> nn.Sequential:
    """Conv2d → BatchNorm2d → ReLU helper."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, **kwargs),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ── Stem ──────────────────────────────────────────────────────────────────────

class StemBlock(nn.Module):
    """
    Pure-conv stem: (3, 299, 299) → (384, 34, 34).

    Two parallel split-and-concat stages reduce spatial size while
    doubling channel depth before the Inception blocks.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        # Shared entry path: stride-2 → 64-ch feature map
        self.entry = nn.Sequential(
            _conv_bn_relu(in_channels, 32, kernel_size=3, stride=2),
            _conv_bn_relu(32, 32, kernel_size=3),
            nn.Conv2d(32, 64, kernel_size=3),   # kept without BN/ReLU to feed two branches
        )

        # First split: MaxPool vs stride-2 conv  → concat 160
        self.pool_a   = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv_a   = _conv_bn_relu(64, 96, kernel_size=3, stride=2)

        # Second split: two branches on 160-ch feature map → concat 192
        self.branch_b = nn.Sequential(
            _conv_bn_relu(160, 64, kernel_size=1),
            _conv_bn_relu(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            _conv_bn_relu(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            _conv_bn_relu(64, 96, kernel_size=3),
        )
        self.branch_c = nn.Sequential(
            _conv_bn_relu(160, 64, kernel_size=1),
            _conv_bn_relu(64, 96, kernel_size=3),
        )

        # Final split: stride-2 conv vs MaxPool → concat 384
        self.conv_out = _conv_bn_relu(192, 192, kernel_size=3, stride=2)
        self.pool_out = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn_out   = nn.BatchNorm2d(384)
        self.relu     = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.entry(x)                                     # (B, 64, H, W)
        x   = torch.cat([self.conv_a(x), self.pool_a(x)], 1)  # 96 + 64 = 160
        x   = torch.cat([self.branch_b(x), self.branch_c(x)], 1)  # 96 + 96 = 192
        out = torch.cat([self.conv_out(x), self.pool_out(x)], 1)   # 192 + 192 = 384
        return self.relu(self.bn_out(out))


# ── Inception-ResNet-A ────────────────────────────────────────────────────────

class InceptionResNetA(nn.Module):
    """
    Inception-ResNet-A block. Repeated 5×.
    in/out channels: 384.

    Three branches produce 32, 32, and 64 channels (128 total), then a
    linear 1×1 projection restores the 384-channel residual width.
    """

    def __init__(self, in_channels: int = 384, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.branch_1x1 = _conv_bn_relu(in_channels, 32, kernel_size=1)
        self.branch_3x3 = nn.Sequential(
            _conv_bn_relu(in_channels, 32, kernel_size=1),
            _conv_bn_relu(32, 32, kernel_size=3, padding=1),
        )
        self.branch_double_3x3 = nn.Sequential(
            _conv_bn_relu(in_channels, 32, kernel_size=1),
            _conv_bn_relu(32, 48, kernel_size=3, padding=1),
            _conv_bn_relu(48, 64, kernel_size=3, padding=1),
        )

        # Keep the expansion linear and scale it before the residual sum.
        self.proj = nn.Conv2d(128, in_channels, kernel_size=1, bias=False)
        self.residual_scale = residual_scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat([
            self.branch_1x1(x),
            self.branch_3x3(x),
            self.branch_double_3x3(x),
        ], 1)
        return self.relu(x + self.residual_scale * self.proj(mixed))


# ── Reduction-A ───────────────────────────────────────────────────────────────

class ReductionA(nn.Module):
    """
    Reduction-A: 384 → 1152.

    Branch A: 1×1 → 3×3 → 3×3/s2 → 384
    Branch B: 3×3/s2 → 384
    Branch C: MaxPool/s2 → 384
    Total: 384 + 384 + 384 = 1152
    """

    def __init__(self, in_channels: int = 384) -> None:
        super().__init__()
        self.branch_a = nn.Sequential(
            _conv_bn_relu(in_channels, 256, kernel_size=1),
            _conv_bn_relu(256, 256, kernel_size=3),
            _conv_bn_relu(256, 384, kernel_size=3, stride=2, padding=1),
        )
        self.branch_b   = _conv_bn_relu(in_channels, 384, kernel_size=3, stride=2)
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.bn   = nn.BatchNorm2d(1152)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([
            self.branch_a(x),
            self.branch_b(x),
            self.branch_pool(x),
        ], 1)
        return self.relu(self.bn(out))


# ── Inception-ResNet-B ────────────────────────────────────────────────────────

class InceptionResNetB(nn.Module):
    """
    Inception-ResNet-B block. Repeated 10×.
    in/out channels: 1152.

    Branch A: 1×1 → (1×7, 7×1) → 192
    Branch B: 1×1 → 192
    Proj: concat(384) → 1×1 → 1152
    """

    def __init__(self, in_channels: int = 1152, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.branch_a = nn.Sequential(
            _conv_bn_relu(in_channels, 128, kernel_size=1),
            _conv_bn_relu(128, 160, kernel_size=(1, 7), padding=(0, 3)),
            _conv_bn_relu(160, 192, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.branch_b = _conv_bn_relu(in_channels, 192, kernel_size=1)

        # Linear projection: 192+192=384 → in_channels
        self.proj = nn.Conv2d(384, in_channels, kernel_size=1, bias=False)
        self.residual_scale = residual_scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat([self.branch_a(x), self.branch_b(x)], 1)
        return self.relu(x + self.residual_scale * self.proj(mixed))


# ── Reduction-B ───────────────────────────────────────────────────────────────

class ReductionB(nn.Module):
    """
    Reduction-B: 1152 → 2144.

    Branch A: 1×1 → 3×3 → 3×3/s2 → 320
    Branch B: 1×1 → 3×3/s2       → 288
    Branch C: 1×1 → 3×3/s2       → 384
    Branch D: MaxPool/s2          → 1152
    Total: 320 + 288 + 384 + 1152 = 2144
    """

    def __init__(self, in_channels: int = 1152) -> None:
        super().__init__()
        self.branch_a = nn.Sequential(
            _conv_bn_relu(in_channels, 256, kernel_size=1),
            _conv_bn_relu(256, 288, kernel_size=3, padding=1),
            _conv_bn_relu(288, 320, kernel_size=3, stride=2, padding=1),
        )
        self.branch_b = nn.Sequential(
            _conv_bn_relu(in_channels, 256, kernel_size=1),
            _conv_bn_relu(256, 288, kernel_size=3, stride=2, padding=1),
        )
        self.branch_c = nn.Sequential(
            _conv_bn_relu(in_channels, 256, kernel_size=1),
            _conv_bn_relu(256, 384, kernel_size=3, stride=2, padding=1),
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn   = nn.BatchNorm2d(2144)   # 320 + 288 + 384 + 1152
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([
            self.branch_a(x),
            self.branch_b(x),
            self.branch_c(x),
            self.branch_pool(x),
        ], 1)
        return self.relu(self.bn(out))


# ── Inception-ResNet-C ────────────────────────────────────────────────────────

class InceptionResNetC(nn.Module):
    """
    Inception-ResNet-C block. Repeated 5×.
    in/out channels: 2144.

    Branch A: 1×1 → (1×3, 3×1) → 256
    Branch B: 1×1 → 192
    Proj: concat(448) → 1×1 → 2144
    """

    def __init__(self, in_channels: int = 2144, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.branch_a = nn.Sequential(
            _conv_bn_relu(in_channels, 192, kernel_size=1),
            _conv_bn_relu(192, 224, kernel_size=(1, 3), padding=(0, 1)),
            _conv_bn_relu(224, 256, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.branch_b = _conv_bn_relu(in_channels, 192, kernel_size=1)

        # Linear projection: 256+192=448 → in_channels
        self.proj = nn.Conv2d(448, in_channels, kernel_size=1, bias=False)
        self.residual_scale = residual_scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat([self.branch_a(x), self.branch_b(x)], 1)
        return self.relu(x + self.residual_scale * self.proj(mixed))
