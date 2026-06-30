"""Per-pixel spectral MLP classifier (Gursch et al. 2026 MLP baseline).

Classifies the center pixel's 437-band spectrum with a three-layer MLP
``C -> hidden1 -> hidden2 -> num_classes`` (ReLU on the hidden layers). Only the patch center is
used, so it ignores spatial context; with ``patch_size=1`` the patch *is* the pixel spectrum.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from cuvis_ai_inspecscrap.node.classification.base import PatchClassifierBase


class SpectralMLPClassifier(PatchClassifierBase):
    """Three-layer per-pixel spectral MLP over the patch center pixel."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden1=hidden1,
            hidden2=hidden2,
            dropout=dropout,
            **kwargs,
        )
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        layers: list[nn.Module] = [nn.Linear(in_channels, hidden1), nn.ReLU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden1, hidden2), nn.ReLU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden2, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, patches: Tensor, **_: Any) -> dict[str, Tensor]:
        """Classify each patch's center pixel: ``[N,P,P,C]`` -> ``logits [N,num_classes]``."""
        p = patches.shape[1]
        center = patches[:, p // 2, p // 2, :]  # [N, C]
        return {"logits": self.net(center)}


__all__ = ["SpectralMLPClassifier"]
