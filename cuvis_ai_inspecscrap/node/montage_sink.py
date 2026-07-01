"""Sink node that assembles captioned montage columns into one side-by-side report image.

The viz pipeline (``ClassMapToRGB`` -> ``LabelOverlay`` -> ``TitleOverlay``) prepares one captioned
panel per montage column, streamed as a batch by a DataModule and run through the
:class:`~cuvis_ai_core.training.predictor.Predictor`. Concatenating those panels side by side and
appending a class legend is a **fan-in over the whole set**, which does not fit the per-batch node
contract, so it lives here in a sink: ``forward`` accumulates each column's panel (and its
colourised label map for the legend), keyed by ``column_index``, and ``close`` stitches them in
column order, appends the legend, and exposes the result via :attr:`montage`.

It mirrors :class:`~cuvis_ai_inspecscrap.node.class_map_accumulator.ClassMapAccumulator`: the
DataModule fans the columns out, this node fans them back into one image. ``reset()`` clears state
at the start of a Predictor run; ``close()`` finalises the montage.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_inspecscrap.node.legend import LegendStripNode


def _concat_horizontal(
    images: list[Tensor],
    gap: int,
    bg_color: tuple[float, float, float],
    align: str,
) -> Tensor:
    """Concatenate ``[1, H, W, 3]`` panels left-to-right, padding heights to the tallest.

    Each panel is padded along the height axis to the common (max) height with ``bg_color``,
    placed per ``align`` ("start" = top, "center", "end" = bottom); a ``gap``-wide ``bg_color``
    separator is inserted between panels. Replaces the former catalog ``ImageConcatenator``
    (removed upstream) for the single horizontal configuration this sink uses.
    """
    ref = images[0]
    bg = torch.tensor(bg_color, dtype=ref.dtype, device=ref.device)
    target_h = max(int(img.shape[1]) for img in images)

    def _pad_h(img: Tensor) -> Tensor:
        cur = int(img.shape[1])
        if cur == target_h:
            return img
        extra = target_h - cur
        before = 0 if align == "start" else extra if align == "end" else extra // 2
        b, _, w, _ = img.shape
        canvas = bg.view(1, 1, 1, 3).expand(b, target_h, w, 3).clone()
        canvas[:, before : before + cur, :, :] = img
        return canvas

    padded = [_pad_h(img) for img in images]
    if gap <= 0:
        return torch.cat(padded, dim=2)
    strip = bg.view(1, 1, 1, 3).expand(ref.shape[0], target_h, gap, 3)
    pieces: list[Tensor] = []
    for k, img in enumerate(padded):
        if k > 0:
            pieces.append(strip)
        pieces.append(img)
    return torch.cat(pieces, dim=2)


class MontageColumnSink(Node):
    """Stitch captioned montage columns into one side-by-side image with a legend (sink)."""

    _category = NodeCategory.SINK
    _tags = frozenset({NodeTag.IMAGE, NodeTag.RGB, NodeTag.POSTPROCESSING})

    INPUT_SPECS = {
        "frame": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Captioned column panels [B, H, W, 3] in [0, 1]",
        ),
        "column_index": PortSpec(
            dtype=torch.int64, shape=(-1,), description="Column position per panel [B]"
        ),
        "label_rgb": PortSpec(
            dtype=torch.Tensor,
            shape=(-1, -1, -1, 3),
            description="Optional colourised label map per panel [B, H, W, 3]; drives legend counts",
            optional=True,
        ),
    }
    OUTPUT_SPECS: dict[str, PortSpec] = {}  # sink node

    def __init__(
        self,
        labelmap_path: str,
        gap: int = 8,
        bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        align: str = "center",
        legend_n_columns: int = 7,
        **kwargs: Any,
    ) -> None:
        if gap < 0:
            raise ValueError(f"gap must be >= 0; got {gap}")
        super().__init__(
            labelmap_path=labelmap_path,
            gap=gap,
            bg_color=list(bg_color),
            align=align,
            legend_n_columns=legend_n_columns,
            **kwargs,
        )
        self.labelmap_path = str(labelmap_path)
        self.gap = int(gap)
        self.bg_color = tuple(float(c) for c in bg_color)
        self.align = align
        self.legend_n_columns = int(legend_n_columns)
        self._panels: dict[int, Tensor] = {}
        self._labels: dict[int, Tensor] = {}
        self._montage: Tensor | None = None

    def reset(self) -> None:
        """Clear accumulated columns before a new montage run (called by the Predictor)."""
        self._panels = {}
        self._labels = {}
        self._montage = None

    @torch.no_grad()
    def forward(
        self,
        frame: Tensor,
        column_index: Tensor,
        label_rgb: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Stash each captioned panel (and its label map) keyed by its column position."""
        for i, col in enumerate(column_index.tolist()):
            self._panels[int(col)] = frame[i].detach().cpu()
            if label_rgb is not None:
                self._labels[int(col)] = label_rgb[i].detach().cpu()
        return {}

    @property
    def montage(self) -> Tensor | None:
        """The finished montage ``[1, H', W', 3]`` in ``[0, 1]`` (``None`` until ``close``)."""
        return None if self._montage is None else self._montage.clone()

    def close(self) -> None:
        """Concatenate the columns in order and append the class legend below them."""
        if not self._panels:
            return
        order = sorted(self._panels)
        images = [self._panels[i].unsqueeze(0) for i in order]  # [1, H, W, 3] per column
        grid = _concat_horizontal(images, gap=self.gap, bg_color=self.bg_color, align=self.align)
        if self._labels:
            legend_col = max(self._labels)  # the rightmost mapped column drives the counts
            legend = LegendStripNode(
                name="_montage_legend",
                labelmap_path=self.labelmap_path,
                n_columns=self.legend_n_columns,
            )
            grid = legend.forward(frame=grid, label_rgb=self._labels[legend_col].unsqueeze(0))[
                "frame"
            ]
        self._montage = grid


__all__ = ["MontageColumnSink"]
