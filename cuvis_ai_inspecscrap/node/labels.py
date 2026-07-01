"""Map colorized PNG label masks to integer class-index targets.

The InSpecScrap PNG masks encode the material class as an RGB colour (see
``LabelMap.txt``). Supervised training needs an integer class-index map instead, merged
from the dataset's 29 colours down to the paper's 14 classes (Gursch et al. 2026).

``RgbLabelToClassIndex`` reads ``LabelMap.txt`` (``name:r,g,b::`` rows), applies a
name -> paper-class mapping, and emits ``targets`` ``[B, H, W]`` int64. It is **fail
closed**: a colour that two source classes share and that would map to two different
paper classes raises at construction unless it is explicitly forced to ``ignore`` (the
``(255, 204, 51)`` ``Me-LightRusty`` / ``Plastic_Packaging`` collision is, by default),
and a pixel colour absent from ``LabelMap.txt`` maps to ``ignore_index`` rather than to
a silently wrong class.

The 29 -> 14 merge here is **provisional**, pending the original per-pixel class ids
from JOANNEUM RESEARCH; it is captured so the label-sanity baseline is reviewable.
"""

from __future__ import annotations

from pathlib import Path

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

#: Paper's 14 material classes, in a fixed order (class id = position). Macro metrics
#: average over classes, so the order is for reproducibility only.
PAPER_CLASSES: tuple[str, ...] = (
    "steel",
    "aluminium",
    "copper",
    "dark_rust_metal",
    "light_rust_metal",
    "painted_metal",
    "can",
    "stone",
    "wood",
    "painted_wood",
    "plastic",
    "rubber",
    "styropor",
    "fabric",
)

#: Provisional mapping from LabelMap.txt source names to a paper class (or ``None`` =
#: ignore). Zero-pixel classes in this release are mapped to a best-guess paper class or
#: ignored; either way they do not affect training. Pending JOANNEUM sign-off.
DEFAULT_SOURCE_TO_PAPER: dict[str, str | None] = {
    "Background": None,
    "background": None,
    "Me-Iron": "steel",
    "Me-Shiny": "steel",
    "Me-StainlessSteel": "steel",
    "MH-Galvanized": "steel",
    "MH-Tinplated": "steel",
    "Me-Package": None,
    "Me-Aluminium": "aluminium",
    "Me-Copper": "copper",
    "ME-Copper-Cased": "copper",
    "Me-DarkRusty": "dark_rust_metal",
    "Me-RustyContam": "dark_rust_metal",
    "Me-LightRusty": "light_rust_metal",  # shares (255,204,51) -> resolved to ignore below
    "ME-Painted": "painted_metal",
    "Can": "can",
    "In-Stone": "stone",
    "Wood": "wood",
    "WO-Painted": "painted_wood",
    "Plastic": "plastic",
    "Cable_Plastic": "plastic",
    "Plastic_Packaging": "plastic",  # shares (255,204,51) -> resolved to ignore below
    "Rubber": "rubber",
    "Styropor": "styropor",
    "Fabric": "fabric",
    # Not in the paper's 14 classes (and zero-pixel in this release): ignored.
    "Glass": None,
    "In-Ceramic": None,
    "Paper": None,
    "Me-Other": None,
    "MH-Dust": None,
}

#: Colours forced to ``ignore`` regardless of name, resolving the documented collision.
DEFAULT_IGNORE_COLORS: tuple[tuple[int, int, int], ...] = ((255, 204, 51),)


def _parse_labelmap(path: Path) -> dict[str, tuple[int, int, int]]:
    """Parse a ``name:r,g,b::`` LabelMap.txt into ``{name: (r, g, b)}``."""
    colors: dict[str, tuple[int, int, int]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        rgb = tuple(int(v) for v in parts[1].split(","))
        if len(rgb) != 3:
            raise ValueError(f"LabelMap row {name!r} has a non-RGB colour: {parts[1]!r}")
        colors[name] = (rgb[0], rgb[1], rgb[2])
    return colors


def _rgb_code(r: int, g: int, b: int) -> int:
    """Pack an RGB triple into one int for fast pixel lookup."""
    return (r << 16) | (g << 8) | b


class RgbLabelToClassIndex(Node):
    """Map a colorized RGB label map to integer paper-class targets (fail-closed)."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.CLASSIFICATION, NodeTag.PREPROCESSING})

    INPUT_SPECS = {
        "label_rgb": PortSpec(
            dtype=torch.Tensor,
            shape=(-1, -1, -1, 3),
            description="Colorized label map [B, H, W, 3], uint8 [0,255] or float32 [0,1]",
        ),
    }
    OUTPUT_SPECS = {
        "targets": PortSpec(
            dtype=torch.int64,
            shape=(-1, -1, -1),
            description="Per-pixel class index [B, H, W]; background/unknown = ignore_index",
        ),
    }

    def __init__(
        self,
        labelmap_path: str,
        source_to_class: dict[str, str | None] | None = None,
        paper_classes: list[str] | None = None,
        ignore_colors: list[list[int]] | None = None,
        ignore_index: int = -100,
        name: str | None = None,
        execution_stages: set[ExecutionStage] | None = None,
    ) -> None:
        classes = tuple(paper_classes) if paper_classes is not None else PAPER_CLASSES
        mapping = dict(source_to_class) if source_to_class is not None else DEFAULT_SOURCE_TO_PAPER
        ignore_rgb = (
            {tuple(c) for c in ignore_colors}
            if ignore_colors is not None
            else set(DEFAULT_IGNORE_COLORS)
        )
        super().__init__(
            labelmap_path=str(labelmap_path),
            source_to_class=mapping,
            paper_classes=list(classes),
            ignore_colors=[list(c) for c in sorted(ignore_rgb)],
            ignore_index=ignore_index,
            name=name,
            execution_stages=execution_stages,
        )
        self.num_classes = len(classes)
        self.ignore_index = int(ignore_index)
        self._class_to_id = {c: i for i, c in enumerate(classes)}

        colors = _parse_labelmap(Path(labelmap_path))
        ignore_codes = {_rgb_code(*c) for c in ignore_rgb}

        # code -> target id, fail closed on a colour two names map to differently.
        code_to_id: dict[int, int] = {}
        code_to_name: dict[int, str] = {}
        for sname, rgb in colors.items():
            code = _rgb_code(*rgb)
            if code in ignore_codes:
                target = self.ignore_index
            else:
                cls = mapping.get(sname)
                target = self._class_to_id[cls] if cls in self._class_to_id else self.ignore_index
            if code in code_to_id and code_to_id[code] != target:
                raise ValueError(
                    f"colour {rgb} is shared by {code_to_name[code]!r} (->{code_to_id[code]}) "
                    f"and {sname!r} (->{target}); add it to ignore_colors to resolve."
                )
            code_to_id[code] = target
            code_to_name[code] = sname

        codes = sorted(code_to_id)
        self.register_buffer(
            "_palette_codes", torch.tensor(codes, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "_palette_ids",
            torch.tensor([code_to_id[c] for c in codes], dtype=torch.int64),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, label_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map ``label_rgb`` [B,H,W,3] to ``targets`` [B,H,W] int64 (unknown -> ignore_index)."""
        rgb = label_rgb
        if torch.is_floating_point(rgb):
            rgb = (rgb * 255.0).round()
        rgb = rgb.to(torch.int64)
        code = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]  # [B,H,W]
        targets = torch.full(code.shape, self.ignore_index, dtype=torch.int64, device=code.device)
        palette_codes = self._palette_codes.to(code.device)
        palette_ids = self._palette_ids.to(code.device)
        for pcode, pid in zip(palette_codes.tolist(), palette_ids.tolist(), strict=False):
            targets[code == pcode] = pid
        return {"targets": targets}
