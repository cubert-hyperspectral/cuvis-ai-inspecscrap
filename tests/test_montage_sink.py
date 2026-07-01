"""Tests for the Predictor-driven montage path: caption port, columns DataModule, sink."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from cuvis_ai_inspecscrap.data.montage_column_datamodule import (
    MontageColumnDataModule,
    _MontageColumnDataset,
)
from cuvis_ai_inspecscrap.node.montage_sink import MontageColumnSink

pytestmark = pytest.mark.unit

# Two foreground classes plus background; enough for the legend strip to render.
_LABELMAP = "Background:0,0,0::\nMe-Iron:10,20,30::\nMe-Aluminium:40,50,60::\n"


def _labelmap(tmp_path) -> str:
    p = tmp_path / "LabelMap.txt"
    p.write_text(_LABELMAP, encoding="utf-8")
    return str(p)


# --- MontageColumnDataModule -------------------------------------------------


def test_montage_dataset_none_map_becomes_background():
    rgb = torch.rand(6, 8, 3)
    ds = _MontageColumnDataset(
        [("plain", rgb, None), ("pred", rgb, torch.zeros(6, 8, dtype=torch.long))],
        background_value=-1,
    )
    plain = ds[0]
    assert plain["class_map"].shape == (6, 8)
    assert torch.all(plain["class_map"] == -1)  # the None column is all background
    assert plain["caption"] == "plain" and plain["column_index"] == 0
    assert ds[1]["column_index"] == 1


def test_montage_datamodule_batches_all_columns_in_order():
    rgb = torch.rand(6, 8, 3)
    cols = [("a", rgb, None), ("b", rgb, torch.zeros(6, 8, dtype=torch.long)), ("c", rgb, None)]
    dm = MontageColumnDataModule(columns=cols)
    dm.setup(stage="predict")
    loader = dm.predict_dataloader()
    assert isinstance(loader, DataLoader)
    batch = next(iter(loader))
    assert batch["rgb_image"].shape == (3, 6, 8, 3)
    assert batch["class_map"].shape == (3, 6, 8)
    assert batch["caption"] == ["a", "b", "c"]  # default collate keeps strings ordered
    assert batch["column_index"].tolist() == [0, 1, 2]


# --- MontageColumnSink -------------------------------------------------------


def test_montage_sink_concatenates_columns_in_order(tmp_path):
    sink = MontageColumnSink(labelmap_path=_labelmap(tmp_path), gap=8, legend_n_columns=7)
    sink.reset()
    panels = torch.stack([torch.full((24, 16, 3), v) for v in (0.2, 0.5, 0.8)])  # [3,24,16,3]
    labels = torch.zeros((3, 24, 16, 3), dtype=torch.uint8)
    # Feed columns out of order to prove the sink sorts by column_index.
    sink.forward(frame=panels, column_index=torch.tensor([2, 0, 1]), label_rgb=labels)
    assert sink.montage is None  # not finalized until close()
    sink.close()

    montage = sink.montage
    assert montage is not None
    n_rows = 1  # two classes wrap into a single legend row at n_columns=7
    legend_h = n_rows * 22 + 2 * 6
    assert montage.shape == (1, 24 + legend_h, 16 * 3 + 8 * 2, 3)
    # Column order restored: the first panel (col 0) is the 0.5 panel, not the 0.2 one fed first.
    assert torch.allclose(montage[0, 0, 0], torch.full((3,), 0.5), atol=1e-4)


def test_montage_sink_reset_clears_state(tmp_path):
    sink = MontageColumnSink(labelmap_path=_labelmap(tmp_path))
    sink.forward(
        frame=torch.zeros((1, 8, 8, 3)),
        column_index=torch.tensor([0]),
        label_rgb=torch.zeros((1, 8, 8, 3), dtype=torch.uint8),
    )
    sink.close()
    assert sink.montage is not None
    sink.reset()
    assert sink.montage is None
    sink.close()  # nothing accumulated -> stays None, no error
    assert sink.montage is None
