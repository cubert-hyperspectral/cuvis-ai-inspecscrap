"""Sink node that rebuilds dense per-pixel class maps from chunked patch predictions.

Dense inference runs the classifier over a frame's patches in batches (the DataLoader ``batch_size``
is the chunk, so memory stays bounded). Each patch carries its provenance ``(frame_id, y, x)`` plus
the source frame ``height``/``width``. This sink consumes the per-batch ``logits`` together with that
provenance, takes the argmax, and scatters the predicted class into the right pixel of a per-frame
map. After a :class:`~cuvis_ai_core.training.predictor.Predictor` run, the finished maps are read
from :attr:`class_maps`.

It is the inverse of the patchify step: ``DensePatchDataModule`` tiles frames into patches keyed by
coordinate; this node stitches the predictions back into ``[H, W]`` class maps. ``reset()`` clears
state at the start of a run and ``close()`` finalizes it, matching the Predictor lifecycle.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class ClassMapAccumulator(Node):
    """Scatter chunked patch predictions back into per-frame ``[H, W]`` class maps (sink)."""

    _category = NodeCategory.SINK
    _tags = frozenset({NodeTag.MASK, NodeTag.CLASSIFICATION, NodeTag.POSTPROCESSING})

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Per-patch class logits [N, num_classes]",
        ),
        "frame_id": PortSpec(
            dtype=torch.int64, shape=(-1,), description="Source frame id per patch [N]"
        ),
        "y": PortSpec(dtype=torch.int64, shape=(-1,), description="Pixel row per patch [N]"),
        "x": PortSpec(dtype=torch.int64, shape=(-1,), description="Pixel column per patch [N]"),
        "height": PortSpec(
            dtype=torch.int64, shape=(-1,), description="Source frame height per patch [N]"
        ),
        "width": PortSpec(
            dtype=torch.int64, shape=(-1,), description="Source frame width per patch [N]"
        ),
    }
    OUTPUT_SPECS: dict[str, PortSpec] = {}  # sink node

    def __init__(self, background_value: int = -1, **kwargs: Any) -> None:
        super().__init__(background_value=background_value, **kwargs)
        self.background_value = int(background_value)
        self._maps: dict[int, Tensor] = {}

    def reset(self) -> None:
        """Clear accumulated maps before a new prediction run (called by the Predictor)."""
        self._maps = {}

    @torch.no_grad()
    def forward(
        self,
        logits: Tensor,
        frame_id: Tensor,
        y: Tensor,
        x: Tensor,
        height: Tensor,
        width: Tensor,
        **_: Any,
    ) -> dict[str, Any]:
        """Argmax the batch's logits and scatter each prediction into its frame's class map."""
        preds = logits.argmax(dim=-1).to(torch.long).cpu()
        fid_t, y_t, x_t = frame_id.cpu(), y.cpu(), x.cpu()
        h_t, w_t = height.cpu(), width.cpu()
        for fid in torch.unique(fid_t).tolist():
            sel = fid_t == fid
            if fid not in self._maps:
                h, w = int(h_t[sel][0]), int(w_t[sel][0])
                self._maps[int(fid)] = torch.full((h, w), self.background_value, dtype=torch.long)
            self._maps[int(fid)][y_t[sel], x_t[sel]] = preds[sel]
        return {}

    @property
    def class_maps(self) -> dict[int, Tensor]:
        """Finished per-frame class maps ``{frame_id: [H, W] int64}`` (background = ``background_value``)."""
        return {fid: m.clone() for fid, m in self._maps.items()}

    def close(self) -> None:
        """No external resource to release; maps stay available via ``class_maps``."""


__all__ = ["ClassMapAccumulator"]
