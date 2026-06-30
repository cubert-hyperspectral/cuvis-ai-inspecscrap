"""3D-CNN spectral-spatial patch classifier (Gursch et al. 2026, best model: 76.47%).

Treats each ``[P, P, C]`` patch as a single-channel 3D volume ``[1, C, P, P]`` (depth = spectral)
and runs three Conv3d-BN-ReLU blocks (channels 8/16/32, spectral kernels 7/5/3) with spectral
max-pooling, then collapses the spectral axis to a fixed size, folds it into the channel dim, runs a
Conv2d block, and a small FC head -> ``logits [N, num_classes]``. Patches are small (7x7), so memory
is bounded; the spectral axis is reduced adaptively so the head size is independent of band count.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from cuvis_ai_inspecscrap.node.classification.base import PatchClassifierBase


class SpectralSpatialCNN3D(PatchClassifierBase):
    """Spectral-spatial 3D-CNN over a patch, classifying the center pixel."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv3d_channels: tuple[int, int, int] = (8, 16, 32),
        conv2d_channels: int = 64,
        spectral_keep: int = 4,
        hidden: tuple[int, int] = (256, 128),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            conv3d_channels=list(conv3d_channels),
            conv2d_channels=conv2d_channels,
            spectral_keep=spectral_keep,
            hidden=list(hidden),
            **kwargs,
        )
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        c1, c2, c3 = conv3d_channels
        self.features3d = nn.Sequential(
            nn.Conv3d(1, c1, (7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(c1),
            nn.ReLU(),
            nn.MaxPool3d((2, 1, 1)),
            nn.Conv3d(c1, c2, (5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(c2),
            nn.ReLU(),
            nn.MaxPool3d((2, 1, 1)),
            nn.Conv3d(c2, c3, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((spectral_keep, None, None)),  # fix spectral, keep spatial P x P
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(c3 * spectral_keep, conv2d_channels, 3, padding=1),
            nn.BatchNorm2d(conv2d_channels),
            nn.ReLU(),
        )
        h1, h2 = hidden
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(conv2d_channels, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_classes),
        )

    def forward(self, patches: Tensor, **_: Any) -> dict[str, Tensor]:
        """``[N,P,P,C]`` -> volume ``[N,1,C,P,P]`` -> 3D conv -> 2D conv -> ``logits [N,num_classes]``."""
        x = patches.permute(0, 3, 1, 2).unsqueeze(1)  # [N, 1, C, P, P]
        x = self.features3d(x)  # [N, c3, spectral_keep, P, P]
        n, cc, d, h, w = x.shape
        x = x.reshape(n, cc * d, h, w)  # fold spectral into channels
        x = self.conv2d(x)
        return {"logits": self.head(x)}


__all__ = ["SpectralSpatialCNN3D"]
