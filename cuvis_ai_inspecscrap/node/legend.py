"""LegendStripNode: append a class-color legend strip below the video frame.

Reads a Cubert-style label map file::

    # label:color_rgb:parts:actions
    Background:0,0,0::
    Cable_Plastic:125,25,77::
    ...

and renders each `<name, (r,g,b)>` entry as a swatch + text label. When the
upstream pipeline also feeds the per-frame `label_rgb` mask through, the legend
appends a connected-component instance count `(N)` per class for the current
frame and dims rows whose count is zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import label as cc_label


def _parse_labelmap(path: Path) -> list[tuple[str, tuple[int, int, int]]]:
    """Parse Cubert LabelMap.txt into `[(name, (r,g,b)), ...]`."""
    entries: list[tuple[str, tuple[int, int, int]]] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        if not name or name.lower() == "background":
            continue
        rgb = parts[1].strip()
        if not rgb:
            continue
        try:
            r, g, b = (int(c) for c in rgb.split(","))
        except ValueError:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        entries.append((name, (r, g, b)))
    return entries


def _count_instances(label_rgb_u8: np.ndarray, color: tuple[int, int, int]) -> int:
    """Connected-component count for one class colour on an H x W x 3 uint8 image."""
    r, g, b = color
    mask = (
        (label_rgb_u8[..., 0] == r)
        & (label_rgb_u8[..., 1] == g)
        & (label_rgb_u8[..., 2] == b)
    )
    if not mask.any():
        return 0
    _, n = cc_label(mask, connectivity=2, return_num=True)
    return int(n)


class LegendStripNode(Node):
    """Append a horizontal class-colour legend strip below the input frame.

    Optionally takes the per-frame `label_rgb` tensor and renders an instance
    count next to each class (connected components on the binary class mask).
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.VIDEO})

    INPUT_SPECS = {
        "frame": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Input frame [B, H, W, 3] in [0, 1]",
        ),
        "label_rgb": PortSpec(
            dtype=torch.Tensor,
            shape=(-1, -1, -1, 3),
            description=(
                "Optional per-frame colorized label map [B, H, W, 3]; uint8 [0, 255] "
                "or float32 [0, 1], used to compute per-class instance counts"
            ),
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "frame": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Frame with legend strip appended [B, H+legend_h, W, 3] in [0, 1]",
        ),
    }

    def __init__(
        self,
        labelmap_path: str,
        n_columns: int = 6,
        tile_height_px: int = 22,
        swatch_width_px: int = 28,
        text_padding_px: int = 6,
        background_color: tuple[float, float, float] = (0.08, 0.08, 0.08),
        text_color: tuple[int, int, int] = (240, 240, 240),
        dim_text_color: tuple[int, int, int] = (110, 110, 110),
        font_size: int = 12,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            labelmap_path=labelmap_path,
            n_columns=n_columns,
            tile_height_px=tile_height_px,
            swatch_width_px=swatch_width_px,
            text_padding_px=text_padding_px,
            background_color=list(background_color),
            text_color=list(text_color),
            dim_text_color=list(dim_text_color),
            font_size=font_size,
            **kwargs,
        )
        self.labelmap_path = Path(labelmap_path)
        self.n_columns = int(n_columns)
        self.tile_height_px = int(tile_height_px)
        self.swatch_width_px = int(swatch_width_px)
        self.text_padding_px = int(text_padding_px)
        self.background_color = background_color
        self.text_color = text_color
        self.dim_text_color = dim_text_color
        self.font_size = int(font_size)

        if not self.labelmap_path.exists():
            raise FileNotFoundError(f"labelmap_path not found: {self.labelmap_path}")
        self._entries = _parse_labelmap(self.labelmap_path)
        if not self._entries:
            raise ValueError(f"No usable entries parsed from {self.labelmap_path}")

        try:
            self._font = ImageFont.truetype("arial.ttf", self.font_size)
        except OSError:
            self._font = ImageFont.load_default()

        n_rows = (len(self._entries) + self.n_columns - 1) // self.n_columns
        self._legend_h = n_rows * self.tile_height_px + 2 * self.text_padding_px

    def _render_strip(self, width: int, counts: list[int] | None) -> torch.Tensor:
        bg = tuple(int(c * 255) for c in self.background_color)
        img = Image.new("RGB", (width, self._legend_h), bg)
        draw = ImageDraw.Draw(img)

        col_w = width // self.n_columns
        for i, (name, rgb) in enumerate(self._entries):
            row = i // self.n_columns
            col = i % self.n_columns
            x0 = col * col_w + self.text_padding_px
            y0 = self.text_padding_px + row * self.tile_height_px
            swatch_x1 = x0 + self.swatch_width_px
            swatch_y1 = y0 + self.tile_height_px - 4
            n = counts[i] if counts is not None else None
            present = n is None or n > 0
            swatch_fill = rgb if present else tuple(c // 3 for c in rgb)
            draw.rectangle(
                [x0, y0, swatch_x1, swatch_y1], fill=swatch_fill, outline=(255, 255, 255)
            )
            label_text = f"{name} ({n})" if n is not None else name
            text_color = self.text_color if present else self.dim_text_color
            text_x = swatch_x1 + self.text_padding_px
            text_y = y0 + max(0, (self.tile_height_px - self.font_size) // 2 - 2)
            draw.text((text_x, text_y), label_text, fill=tuple(text_color), font=self._font)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    def forward(
        self,
        frame: torch.Tensor,
        label_rgb: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        b, h, w, c = frame.shape
        counts: list[int] | None = None
        if label_rgb is not None:
            # Use the first batch element — pipeline always streams batch_size=1. label_rgb is
            # uint8 [0, 255] from the tiff_paired DataModule, or float32 [0, 1] from an upstream node.
            lab = label_rgb[0].detach().cpu()
            if torch.is_floating_point(lab):
                arr_u8 = (lab.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
            else:
                arr_u8 = lab.to(torch.uint8).numpy()
            counts = [_count_instances(arr_u8, color) for _, color in self._entries]

        strip = self._render_strip(w, counts).to(device=frame.device, dtype=frame.dtype)
        strip = strip.unsqueeze(0).expand(b, -1, -1, -1)
        return {"frame": torch.cat([frame, strip], dim=1).clamp(0.0, 1.0)}
