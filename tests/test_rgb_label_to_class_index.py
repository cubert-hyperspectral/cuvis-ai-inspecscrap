"""Tests for RgbLabelToClassIndex: color->class-index, fail-closed collisions/unknowns."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cuvis_ai_inspecscrap.node.labels import DEFAULT_IGNORE_COLORS, RgbLabelToClassIndex

pytestmark = pytest.mark.unit


def _write_labelmap(path: Path, rows: list[tuple[str, tuple[int, int, int]]]) -> Path:
    lines = ["# label:color_rgb:parts:actions"]
    lines += [f"{name}:{r},{g},{b}::" for name, (r, g, b) in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_maps_known_colors_collision_and_unknown(tmp_path):
    lm = _write_labelmap(
        tmp_path / "LabelMap.txt",
        [
            ("Background", (0, 0, 0)),
            ("Me-Iron", (242, 239, 244)),
            ("Me-Copper", (133, 75, 20)),
            ("Me-LightRusty", (255, 204, 51)),
            ("Plastic_Packaging", (255, 204, 51)),  # collision
            ("background", (0, 0, 0)),  # duplicate
        ],
    )
    node = RgbLabelToClassIndex(
        labelmap_path=str(lm),
        source_to_class={
            "Background": None,
            "background": None,
            "Me-Iron": "steel",
            "Me-Copper": "copper",
            "Me-LightRusty": "light_rust_metal",
            "Plastic_Packaging": "plastic",
        },
        paper_classes=["steel", "copper", "light_rust_metal", "plastic"],
        ignore_index=-100,
    )
    # [1, 2, 3] image: iron, copper, collision-color, unknown, background
    label = torch.tensor(
        [[[242, 239, 244], [133, 75, 20], [255, 204, 51]], [[9, 9, 9], [0, 0, 0], [242, 239, 244]]],
        dtype=torch.uint8,
    ).unsqueeze(0)  # [1, 2, 3, 3]
    out = node.forward(label_rgb=label)["targets"]

    assert out.shape == (1, 2, 3)
    assert out.dtype == torch.int64
    assert out[0, 0, 0].item() == 0  # steel
    assert out[0, 0, 1].item() == 1  # copper
    assert out[0, 0, 2].item() == -100  # collision -> ignore (default ignore_colors)
    assert out[0, 1, 0].item() == -100  # unknown -> ignore
    assert out[0, 1, 1].item() == -100  # background -> ignore
    assert out[0, 1, 2].item() == 0  # steel again
    assert node.num_classes == 4


def test_build_time_collision_raises_without_ignore(tmp_path):
    lm = _write_labelmap(
        tmp_path / "LabelMap.txt",
        [("A", (10, 20, 30)), ("B", (10, 20, 30))],  # same color, different classes
    )
    with pytest.raises(ValueError, match="shared"):
        RgbLabelToClassIndex(
            labelmap_path=str(lm),
            source_to_class={"A": "steel", "B": "copper"},
            paper_classes=["steel", "copper"],
            ignore_colors=[],  # do not resolve -> must fail closed
        )


def test_float_input_normalized(tmp_path):
    lm = _write_labelmap(tmp_path / "LabelMap.txt", [("Me-Iron", (255, 0, 0))])
    node = RgbLabelToClassIndex(
        labelmap_path=str(lm),
        source_to_class={"Me-Iron": "steel"},
        paper_classes=["steel"],
    )
    label_f = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32)  # [1,1,1,3] in [0,1]
    out = node.forward(label_rgb=label_f)["targets"]
    assert out[0, 0, 0].item() == 0


def test_real_labelmap_default_mapping_builds(tmp_path):
    """The shipped defaults must build against the real LabelMap.txt (collision resolved)."""
    real = Path("data/HSIMetalScrap/LabelMap.txt")
    if not real.exists():
        pytest.skip("real LabelMap.txt not present")
    node = RgbLabelToClassIndex(labelmap_path=str(real))
    assert node.num_classes == 14
    assert tuple(DEFAULT_IGNORE_COLORS[0]) == (255, 204, 51)
    # Me-Iron (242,239,244) -> steel (id 0); collision color -> ignore.
    label = torch.tensor(
        [[[242, 239, 244], [255, 204, 51], [0, 0, 0]]], dtype=torch.uint8
    ).unsqueeze(0)
    out = node.forward(label_rgb=label)["targets"]
    assert out[0, 0, 0].item() == 0
    assert out[0, 0, 1].item() == -100
    assert out[0, 0, 2].item() == -100
