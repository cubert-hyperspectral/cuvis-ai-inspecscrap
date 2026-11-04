"""Batch-key bridge from the patch data modules into cuvis-ai's ``Standardize`` node.

The patch data modules (``MetalScrapPatchDataModule`` and friends) and ``PatchSampler`` emit a
batch keyed ``patches`` ``[N, P, P, C]``. cuvis-ai's built-in
``cuvis_ai.node.pretreatments.standardize.Standardize`` (``axis="channel"``, ``ddof=0``,
``eps_mode="add"``) reproduces the paper's global per-band z-score exactly (see the pipeline yamls
under ``configs/pipeline/``), but its ports are fixed at ``cube`` in and out, so it cannot sit
directly at the batch entry point. ``PatchesToCube`` is a pure port rename, no math: it reads
``patches`` from the batch and re-emits the same tensor as ``cube``, which feeds the standardizer by
edge; the standardizer's own ``cube`` output then feeds the classifier's ``patches`` input by edge.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class PatchesToCube(Node):
    """Rename the patch data modules' ``patches`` batch key to ``cube`` (no other change)."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING})

    INPUT_SPECS = {
        "patches": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Center-pixel patches [N, P, P, C] from the batch",
        ),
    }
    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Same tensor, renamed for cuvis_ai_builtin's Standardize",
        ),
    }

    def forward(self, patches: Tensor, **_: Any) -> dict[str, Tensor]:
        """Pass the patches tensor through unchanged under the ``cube`` port name."""
        return {"cube": patches}


__all__ = ["PatchesToCube"]
