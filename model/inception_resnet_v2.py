"""
model/inception_resnet_v2.py — Assembles all blocks into the full model.
"""

import torch
import torch.nn as nn

from .blocks import (
    StemBlock,
    InceptionResNetA, ReductionA,
    InceptionResNetB, ReductionB,
    InceptionResNetC,
)


class InceptionResNetV2(nn.Module):
    """
    Inception-ResNet-v2 built from scratch.

    Architecture:
        Stem
        5 × Inception-ResNet-A  (384-ch)
        Reduction-A             (384 → 1152)
        10 × Inception-ResNet-B (1152-ch)
        Reduction-B             (1152 → 2144)
        5 × Inception-ResNet-C  (2144-ch)
        AvgPool → Dropout → FC(2144→1024) → FC(1024→512) → FC(512→num_classes)

    Args:
        in_channels:  Input image channels (default 3).
        num_classes:  Number of output classes (default 257 for Caltech-256).
        dropout:      Dropout probability before the FC layers (default 0.8).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 257,
        dropout: float = 0.8,
    ) -> None:
        super().__init__()

        self.stem = StemBlock(in_channels)

        self.inception_a = nn.Sequential(
            *[InceptionResNetA(in_channels=384) for _ in range(5)]
        )

        self.reduction_a = ReductionA(in_channels=384)

        self.inception_b = nn.Sequential(
            *[InceptionResNetB(in_channels=1152) for _ in range(10)]
        )

        self.reduction_b = ReductionB(in_channels=1152)

        self.inception_c = nn.Sequential(
            *[InceptionResNetC(in_channels=2144) for _ in range(5)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)

        self.fc1     = nn.Linear(2144, 1024)
        self.fc2     = nn.Linear(1024, 512)
        self.fc_out  = nn.Linear(512, num_classes)
        self.relu    = nn.ReLU(inplace=True)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)          # (B, 384, 34, 34)

        x = self.inception_a(x)   # (B, 384, 34, 34)
        x = self.reduction_a(x)   # (B, 1152, 16, 16)

        x = self.inception_b(x)   # (B, 1152, 16, 16)
        x = self.reduction_b(x)   # (B, 2144, 8, 8)

        x = self.inception_c(x)   # (B, 2144, 8, 8)

        x = self.avgpool(x)       # (B, 2144, 1, 1)
        x = torch.flatten(x, 1)  # (B, 2144)
        x = self.dropout(x)

        x = self.relu(self.fc1(x))   # (B, 1024)
        x = self.relu(self.fc2(x))   # (B, 512)
        return self.fc_out(x)         # (B, num_classes)

    # ── Convenience ───────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
