"""Tests for the dense-inference path: lazy DensePatchDataModule + ClassMapAccumulator sink."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from cuvis_ai_inspecscrap.data.dense_patch_datamodule import (
    DensePatchDataModule,
    _DensePatchDataset,
)
from cuvis_ai_inspecscrap.node.class_map_accumulator import ClassMapAccumulator

pytestmark = pytest.mark.unit

# Me-Iron -> steel(0), Me-Aluminium -> aluminium(1); Background -> ignore.
_LABELMAP = "Background:0,0,0::\nMe-Iron:10,20,30::\nMe-Aluminium:40,50,60::\n"


class _FakeFrames(Dataset):
    """Synthetic frames: random cube + a label split into steel / aluminium / background."""

    def __init__(self, n_frames: int, h: int = 6, w: int = 8, c: int = 4) -> None:
        rng = np.random.default_rng(0)
        self._frames = []
        for i in range(n_frames):
            cube = rng.standard_normal((h, w, c)).astype(np.float32)
            label = np.zeros((h, w, 3), dtype=np.uint8)
            label[: h // 2, :] = (10, 20, 30)  # steel
            label[h // 2 :, : w // 2] = (40, 50, 60)  # aluminium; bottom-right stays background
            self._frames.append({"cube": cube, "label_rgb": label, "stem": f"f{i}"})

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict:
        return self._frames[idx]


def _labelmap(tmp_path) -> str:
    p = tmp_path / "LabelMap.txt"
    p.write_text(_LABELMAP, encoding="utf-8")
    return str(p)


def _frames_for_dataset(n: int = 1, h: int = 6, w: int = 8, c: int = 4):
    """Build the (cube, targets) frame dicts the _DensePatchDataset consumes directly."""
    rng = np.random.default_rng(1)
    frames = []
    for i in range(n):
        cube = rng.standard_normal((h, w, c)).astype(np.float32)
        targets = torch.full((h, w), -100, dtype=torch.int64)
        targets[: h // 2, :] = 0  # steel
        targets[h // 2 :, : w // 2] = 1  # aluminium; bottom-right ignored
        frames.append({"cube": cube, "targets": targets, "stem": f"f{i}"})
    return frames


# --- _DensePatchDataset ------------------------------------------------------


def test_dense_dataset_len_equals_foreground_count():
    h, w = 6, 8
    frames = _frames_for_dataset(1, h=h, w=w)
    fg = int((frames[0]["targets"] != -100).sum())
    ds = _DensePatchDataset(frames, patch_size=7, ignore_index=-100)
    assert len(ds) == fg


def test_dense_dataset_item_shape_and_provenance():
    frames = _frames_for_dataset(1, h=6, w=8, c=4)
    ds = _DensePatchDataset(frames, patch_size=7, ignore_index=-100)
    item = ds[0]
    assert item["patches"].shape == (7, 7, 4)
    assert item["patches"].dtype == torch.float32
    # First foreground pixel is row-major (0, 0); provenance + frame shape come back.
    assert item["frame_id"] == 0 and item["y"] == 0 and item["x"] == 0
    assert item["height"] == 6 and item["width"] == 8
    assert item["targets"] == 0  # center label of the first foreground pixel (steel)


def test_dense_dataset_is_lazy_not_pre_expanded():
    """The dataset stores only padded cubes + an [N,3] index, never an [N,P,P,C] patch tensor."""
    frames = _frames_for_dataset(1, h=6, w=8)
    ds = _DensePatchDataset(frames, patch_size=7, ignore_index=-100)
    assert ds._items.shape[1] == 3  # (frame_id, y, x) rows only
    assert len(ds._padded) == 1  # one resident padded cube, patches gathered on demand


def test_dense_dataset_center_pixel_matches_cube():
    """The gathered window's center equals the original cube pixel (reflect-pad alignment)."""
    frames = _frames_for_dataset(1, h=6, w=8, c=4)
    cube = torch.from_numpy(frames[0]["cube"])
    ds = _DensePatchDataset(frames, patch_size=7, ignore_index=-100)
    item = ds[5]  # some foreground pixel
    y, x = item["y"], item["x"]
    assert torch.allclose(item["patches"][3, 3], cube[y, x])  # center of a 7x7 patch


# --- DensePatchDataModule ----------------------------------------------------


def test_datamodule_predict_dataloader_batches_with_provenance(tmp_path, monkeypatch):
    dm = DensePatchDataModule(batch_size=16, patch_size=7, labelmap_path=_labelmap(tmp_path))
    monkeypatch.setattr(dm, "_frame_dataset", lambda stage: _FakeFrames(1, h=6, w=8, c=4))
    dm.setup(stage="predict")
    loader = dm.predict_dataloader()
    batch = next(iter(loader))
    assert batch["patches"].shape[1:] == (7, 7, 4)
    for key in ("frame_id", "y", "x", "height", "width", "targets"):
        assert batch[key].shape[0] == batch["patches"].shape[0]
    assert batch["height"][0].item() == 6 and batch["width"][0].item() == 8


# --- ClassMapAccumulator -----------------------------------------------------


def test_accumulator_scatters_argmax_into_map():
    acc = ClassMapAccumulator(background_value=-1)
    acc.reset()
    # 2x2 frame, 3 labelled pixels: (0,0)->2, (0,1)->0, (1,1)->1; (1,0) stays background.
    logits = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    acc.forward(
        logits=logits,
        frame_id=torch.tensor([0, 0, 0]),
        y=torch.tensor([0, 0, 1]),
        x=torch.tensor([0, 1, 1]),
        height=torch.tensor([2, 2, 2]),
        width=torch.tensor([2, 2, 2]),
    )
    m = acc.class_maps[0]
    assert m.shape == (2, 2)
    assert m[0, 0] == 2 and m[0, 1] == 0 and m[1, 1] == 1
    assert m[1, 0] == -1  # untouched pixel stays background


def test_accumulator_reset_and_two_frames_independent():
    acc = ClassMapAccumulator()
    acc.reset()
    acc.forward(
        logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        frame_id=torch.tensor([0, 1]),
        y=torch.tensor([0, 0]),
        x=torch.tensor([0, 0]),
        height=torch.tensor([1, 3]),
        width=torch.tensor([1, 3]),
    )
    maps = acc.class_maps
    assert set(maps) == {0, 1}
    assert maps[0].shape == (1, 1) and maps[1].shape == (3, 3)
    assert maps[0][0, 0] == 0 and maps[1][0, 0] == 1
    acc.reset()
    assert acc.class_maps == {}
