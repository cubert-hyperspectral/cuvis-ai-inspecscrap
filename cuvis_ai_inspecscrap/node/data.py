"""Minimal batch -> graph adapter for the cuvis-ai-dataloader ``tiff_paired`` DataModule.

Channel selectors (e.g. ``FastRGBSelector``) declare ``wavelengths`` as numpy ``int32``, but the
collated DataModule batch delivers torch tensors, and the pipeline validates dtypes strictly. This
node mirrors cuvis-ai's built-in ``CU3SDataNode``: it takes the batch's ``cube`` + ``wavelengths``
and re-emits ``cube`` as float32 and ``wavelengths`` as numpy ``int32`` [C], which the selectors
accept. ``label_rgb`` is consumed straight from the batch by the compositor / legend nodes (their
specs accept the batch's uint8), so it does not pass through here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec


class TiffDataNode(Node):
    """Adapt ``tiff_paired`` batch keys to the dtypes the channel selectors expect."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.HYPERSPECTRAL})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.Tensor,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C] from the DataModule batch",
        ),
        "wavelengths": PortSpec(
            dtype=torch.Tensor,
            shape=(-1, -1),
            description="Per-channel wavelengths [B, C] in nm from the DataModule batch",
        ),
    }
    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C] as float32",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Wavelengths [C] in nm as numpy int32 (channel-selector parity)",
        ),
    }

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: torch.Tensor,
        **_: Any,
    ) -> dict[str, Any]:
        wl = wavelengths[0] if wavelengths.ndim == 2 else wavelengths
        return {
            "cube": cube.to(torch.float32),
            "wavelengths": wl.detach().cpu().numpy().astype(np.int32),
        }
