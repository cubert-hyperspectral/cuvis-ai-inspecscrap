"""Shared base for per-patch HSI classifiers.

All three models (MLP, 2D-CNN, 3D-CNN) take center-pixel patches ``[N, P, P, C]`` and emit
``logits [N, num_classes]`` (P = 1 reduces to the single-pixel spectrum, the paper's MLP input).
The base owns the port specs and the symbolic ``num_classes`` output dim; subclasses set
``self.num_classes`` + ``self.in_channels`` and implement ``forward``. Default ``Node.freeze`` /
``unfreeze`` already toggle ``requires_grad`` on the nn.Module parameters, so no override is needed.
"""

from __future__ import annotations

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec


class PatchClassifierBase(Node):
    """Center-pixel patch classifier: ``patches [N,P,P,C]`` -> ``logits [N,num_classes]``."""

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.CLASSIFICATION, NodeTag.LEARNABLE, NodeTag.TORCH}
    )

    INPUT_SPECS = {
        "patches": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Center-pixel patches [N, P, P, C] (P=1 for the spectral MLP)",
        ),
    }
    OUTPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, "num_classes"),
            description="Per-patch class logits [N, num_classes]",
        ),
    }


__all__ = ["PatchClassifierBase"]
