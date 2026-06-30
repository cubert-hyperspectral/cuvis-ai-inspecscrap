"""Global per-channel (per-band) standardization for patches (Gursch et al. 2026 preprocessing).

The paper z-scores each spectral band by the **global** training mean/std, not per-sample. The core
``ZScoreNormalizer`` is per-sample (it standardizes each frame by its own spatial stats), so it does
not reproduce that. ``PerChannelStandardizer`` fits per-band mean/std once in Phase 1
(``statistical_initialization``, driven by ``StatisticalTrainer``), stores them as frozen buffers,
and applies ``(x - mean) / (std + eps)`` along the channel axis at every stage. It sits first in the
classifier pipeline: the batch ``patches`` feed it, and its ``normalized`` output feeds the classifier.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class PerChannelStandardizer(Node):
    """Standardize patches per spectral band with global training stats fitted in Phase 1."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.NORMALIZATION, NodeTag.PREPROCESSING, NodeTag.HYPERSPECTRAL})

    INPUT_SPECS = {
        "patches": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Center-pixel patches [N, P, P, C]",
        ),
    }
    OUTPUT_SPECS = {
        "normalized": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-band standardized patches [N, P, P, C]",
        ),
    }

    def __init__(self, in_channels: int, eps: float = 1e-6, **kwargs: Any) -> None:
        self.in_channels = int(in_channels)
        self.eps = float(eps)
        super().__init__(in_channels=in_channels, eps=eps, **kwargs)
        self.register_buffer("mean", torch.zeros(self.in_channels))
        self.register_buffer("std", torch.ones(self.in_channels))
        # Global stats must be fitted before forward; this node always needs Phase 1.
        self._requires_initial_fit_override = True
        self._statistically_initialized = False

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Accumulate per-band mean/std over the training patches (single pass)."""
        c = self.in_channels
        total: Tensor | None = None
        total_sq: Tensor | None = None
        count = 0
        for batch_data in input_stream:
            x = batch_data.get("patches")
            if x is None:
                continue
            flat = x.reshape(-1, c).to(torch.float64)
            if total is None:
                total = torch.zeros(c, dtype=torch.float64, device=flat.device)
                total_sq = torch.zeros(c, dtype=torch.float64, device=flat.device)
            total += flat.sum(dim=0)
            total_sq += (flat * flat).sum(dim=0)
            count += flat.shape[0]
        if count == 0 or total is None:
            raise RuntimeError(
                "PerChannelStandardizer.statistical_initialization() received no patches."
            )
        mean = total / count
        var = (total_sq / count) - mean * mean
        std = var.clamp_min(0.0).sqrt()
        self.mean.copy_(mean.to(self.mean.dtype).cpu())
        self.std.copy_(std.to(self.std.dtype).cpu())
        self._statistically_initialized = True

    def forward(self, patches: Tensor, **_: Any) -> dict[str, Tensor]:
        """Standardize each band: ``(x - mean) / (std + eps)`` along the channel axis."""
        if not self._statistically_initialized:
            raise RuntimeError(
                "PerChannelStandardizer requires statistical_initialization() before forward()."
            )
        mean = self.mean.to(device=patches.device, dtype=patches.dtype)
        std = self.std.to(device=patches.device, dtype=patches.dtype)
        return {"normalized": (patches - mean) / (std + self.eps)}


__all__ = ["PerChannelStandardizer"]
