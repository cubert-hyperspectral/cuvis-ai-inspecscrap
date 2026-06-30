"""Tests for MetalScrapPatchDataModule: patch extraction, cache reuse, batch collation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from cuvis_ai_inspecscrap.data.metal_scrap_patch_datamodule import (
    MetalScrapPatchDataModule,
    _TensorPatchDataset,
)

pytestmark = pytest.mark.unit

# Source names from DEFAULT_SOURCE_TO_PAPER and their paper-class ids: Me-Iron->steel(0),
# Me-Aluminium->aluminium(1). Background->ignore.
_LABELMAP = "Background:0,0,0::\nMe-Iron:10,20,30::\nMe-Aluminium:40,50,60::\n"


class _FakeFrames(Dataset):
    """A few synthetic frames: random cube + a label split into steel / aluminium / background."""

    def __init__(self, n_frames: int, h: int = 8, w: int = 8, c: int = 4) -> None:
        self._frames = []
        rng = np.random.default_rng(0)
        for i in range(n_frames):
            cube = rng.standard_normal((h, w, c)).astype(np.float32)
            label = np.zeros((h, w, 3), dtype=np.uint8)
            label[: h // 2, :] = (10, 20, 30)  # steel
            label[h // 2 :, : w // 2] = (40, 50, 60)  # aluminium
            self._frames.append({"cube": cube, "label_rgb": label, "stem": f"f{i}"})

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict:
        return self._frames[idx]


def _labelmap(tmp_path) -> str:
    p = tmp_path / "LabelMap.txt"
    p.write_text(_LABELMAP, encoding="utf-8")
    return str(p)


def test_tensor_patch_dataset_item_and_collate():
    patches = torch.arange(5 * 3 * 3 * 4, dtype=torch.float32).reshape(5, 3, 3, 4)
    labels = torch.tensor([0, 1, 2, 1, 0])
    ds = _TensorPatchDataset(patches, labels)
    assert len(ds) == 5
    item = ds[1]
    assert item["patches"].shape == (3, 3, 4)
    assert item["targets"] == 1 and isinstance(item["targets"], int)

    batch = next(iter(DataLoader(ds, batch_size=4)))
    assert batch["patches"].shape == (4, 3, 3, 4)
    assert batch["targets"].shape == (4,)
    assert batch["targets"].dtype == torch.int64


def test_build_stage_dataset_train_shapes(tmp_path, monkeypatch):
    dm = MetalScrapPatchDataModule(
        root=str(tmp_path),
        patch_size=3,
        train_samples=12,
        seed=0,
        cache_dir=str(tmp_path / "cache"),
        labelmap_path=_labelmap(tmp_path),
    )
    monkeypatch.setattr(dm, "_frame_dataset", lambda stage: _FakeFrames(2))

    train_ds = dm.build_stage_dataset("train")
    assert len(train_ds) > 0
    item = train_ds[0]
    assert item["patches"].shape == (3, 3, 4)  # P x P x C
    labels = train_ds.labels
    assert labels.dtype == torch.int64
    # Only steel(0) / aluminium(1) are labeled; background is ignored, never sampled.
    assert set(labels.unique().tolist()) <= {0, 1}


def test_eval_stage_is_dense_capped(tmp_path, monkeypatch):
    dm = MetalScrapPatchDataModule(
        root=str(tmp_path),
        patch_size=3,
        eval_cap=5,
        seed=0,
        cache_dir=str(tmp_path / "cache"),
        labelmap_path=_labelmap(tmp_path),
    )
    monkeypatch.setattr(dm, "_frame_dataset", lambda stage: _FakeFrames(2))
    test_ds = dm.build_stage_dataset("test")
    # 2 frames, capped at 5 labeled pixels each (strided), so at most 10 patches.
    assert 0 < len(test_ds) <= 10


def test_cache_is_reused(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    dm = MetalScrapPatchDataModule(
        root=str(tmp_path), patch_size=3, train_samples=8, seed=0,
        cache_dir=str(cache), labelmap_path=_labelmap(tmp_path),
    )
    calls = {"n": 0}

    def _counting(stage):
        calls["n"] += 1
        return _FakeFrames(2)

    monkeypatch.setattr(dm, "_frame_dataset", _counting)
    a = dm.build_stage_dataset("train")
    assert (cache / "train_ps3_seed0_n8.pt").exists()
    assert calls["n"] == 1
    # Second build hits the cache file, so the frame source is not read again.
    b = dm.build_stage_dataset("train")
    assert calls["n"] == 1
    assert torch.equal(a.labels, b.labels)


def test_predict_reuses_test_patches(tmp_path, monkeypatch):
    dm = MetalScrapPatchDataModule(
        root=str(tmp_path), patch_size=3, eval_cap=5, seed=0,
        cache_dir=str(tmp_path / "cache"), labelmap_path=_labelmap(tmp_path),
    )
    monkeypatch.setattr(dm, "_frame_dataset", lambda stage: _FakeFrames(1))
    dm.build_stage_dataset("test")
    # predict resolves to the test cache key; no rebuild.
    assert dm._cache_file("predict") == dm._cache_file("test")
    predict_ds = dm.build_stage_dataset("predict")
    assert len(predict_ds) > 0


def test_validate_params_requires_root():
    with pytest.raises(ValueError, match="root"):
        MetalScrapPatchDataModule.validate_params({})
