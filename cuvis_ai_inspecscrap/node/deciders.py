"""Object-level majority-vote decider (Gursch et al. 2026 inference step).

Given per-pixel class ``predictions`` and an object grouping, relabel every pixel in an object to
that object's majority predicted class. Objects come from an ``instances`` id map, or are derived
from a ``foreground`` mask via connected components (``skimage.measure.label``). This is the paper's
inference-time vote that turns noisy per-pixel predictions into one class per scrap piece, and it
drives the copper-vs-steel confusion read.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class BlobMajorityVote(Node):
    """Relabel each object (instance / connected component) to its majority predicted class."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.CLASSIFICATION, NodeTag.POSTPROCESSING})

    INPUT_SPECS = {
        "predictions": PortSpec(
            dtype=torch.int64, shape=(-1, -1, -1), description="Per-pixel class predictions [B,H,W]"
        ),
        "instances": PortSpec(
            dtype=torch.int64,
            shape=(-1, -1, -1),
            description="Object id per pixel [B,H,W]; ignore_instance is left unchanged (optional)",
            optional=True,
        ),
        "foreground": PortSpec(
            dtype=torch.int64,
            shape=(-1, -1, -1),
            description="Foreground mask [B,H,W] (>0) to derive instances if none given (optional)",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "predictions": PortSpec(
            dtype=torch.int64, shape=(-1, -1, -1), description="Object-voted predictions [B,H,W]"
        ),
    }

    def __init__(
        self,
        num_classes: int | None = None,
        ignore_instance: int = 0,
        connectivity: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            ignore_instance=ignore_instance,
            connectivity=connectivity,
            **kwargs,
        )
        self.num_classes = num_classes
        self.ignore_instance = int(ignore_instance)
        self.connectivity = int(connectivity)

    def _derive_instances(self, foreground_frame: Tensor) -> Tensor:
        """Connected-component instance ids from a foreground mask (background -> ignore_instance)."""
        from skimage.measure import label as cc_label

        fg = (foreground_frame > 0).cpu().numpy()
        labeled = cc_label(fg, connectivity=self.connectivity)  # 0 = background
        inst = torch.as_tensor(labeled.astype("int64"), device=foreground_frame.device)
        if self.ignore_instance != 0:
            # cc_label uses 0 for background; remap so background == ignore_instance.
            inst = torch.where(inst == 0, torch.tensor(self.ignore_instance, device=inst.device), inst)
        return inst

    @torch.no_grad()
    def forward(
        self,
        predictions: Tensor,
        instances: Tensor | None = None,
        foreground: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Vote each object to its majority class; pixels in ``ignore_instance`` stay unchanged."""
        if instances is None and foreground is None:
            raise ValueError("BlobMajorityVote needs either 'instances' or 'foreground'.")
        b = predictions.shape[0]
        k = self.num_classes if self.num_classes is not None else int(predictions.max()) + 1
        out = predictions.clone()
        for i in range(b):
            preds_i = predictions[i]
            inst_i = instances[i] if instances is not None else self._derive_instances(foreground[i])
            for obj in torch.unique(inst_i).tolist():
                if obj == self.ignore_instance:
                    continue
                mask = inst_i == obj
                vals = preds_i[mask]
                majority = int(torch.bincount(vals, minlength=k).argmax())
                out[i][mask] = majority
        return {"predictions": out}


__all__ = ["BlobMajorityVote"]
