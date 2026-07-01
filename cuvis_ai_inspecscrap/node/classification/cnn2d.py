"""2D-CNN spatial-spectral patch classifier (Gursch et al. 2026, 74.06%).

Treats the ``[P, P, C]`` patch as a ``C``-channel image ``[C, P, P]`` and runs three
Conv2d-BN-ReLU blocks (32/64/128, 3x3, same padding), then global average pooling + a linear head
-> ``logits [N, num_classes]``. Spatial context comes from the 3x3 convolutions over the patch.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from cuvis_ai_inspecscrap.node.classification.base import PatchClassifierBase


class SpatialSpectralCNN2D(PatchClassifierBase):
    """Spatial-spectral 2D-CNN over a patch (bands as input channels)."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int] = (32, 64, 128),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            channels=list(channels),
            **kwargs,
        )
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, num_classes))

    def forward(self, patches: Tensor, **_: Any) -> dict[str, Tensor]:
        """``[N,P,P,C]`` -> ``[N,C,P,P]`` -> conv blocks -> ``logits [N,num_classes]``."""
        x = patches.permute(0, 3, 1, 2)  # [N, C, P, P]
        x = self.features(x)
        return {"logits": self.head(x)}


__all__ = ["SpatialSpectralCNN2D"]
