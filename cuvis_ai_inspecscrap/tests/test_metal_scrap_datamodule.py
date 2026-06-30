"""Tests for MetalScrapDataModule: random-frame + dataset-level splits, disjointness, label path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cuvis_ai_inspecscrap.data import MetalScrapDataModule

pytestmark = pytest.mark.unit


def _make_tree(root: Path, layout: dict[str, int]) -> None:
    """Create <root>/DataSetN/images/*.tif (empty) + labels/*.png (tiny RGB)."""
    from PIL import Image

    for ds, n in layout.items():
        images = root / ds / "images"
        labels = root / ds / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(n):
            stem = f"{ds}_{i:02d}"
            (images / f"{stem}.tif").write_bytes(b"")  # content unused when read_cube=False
            Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(labels / f"{stem}.png")


def test_dataset_level_split_is_disjoint(tmp_path):
    _make_tree(tmp_path, {"DataSet0": 6, "DataSet1": 4, "DataSet3": 5})
    dm = MetalScrapDataModule(
        root=str(tmp_path),
        split_mode="dataset",
        test_datasets=["DataSet3"],
        val_fraction=0.2,
        seed=0,
        read_cube=False,
    )
    dm.setup()

    train, val, test = dm.train_ds, dm.val_ds, dm.test_ds
    assert len(test) == 5
    assert {test[i]["dataset"] for i in range(len(test))} == {"DataSet3"}
    assert len(train) + len(val) == 10
    assert len(val) == 2
    train_stems = {train[i]["stem"] for i in range(len(train))}
    val_stems = {val[i]["stem"] for i in range(len(val))}
    test_stems = {test[i]["stem"] for i in range(len(test))}
    assert train_stems.isdisjoint(val_stems)
    assert (train_stems | val_stems).isdisjoint(test_stems)
    assert all(not s.startswith("DataSet3") for s in (train_stems | val_stems))


def test_random_frame_split_mixes_datasets_and_is_disjoint(tmp_path):
    _make_tree(tmp_path, {"DataSet0": 10, "DataSet1": 6, "DataSet3": 4})
    dm = MetalScrapDataModule(
        root=str(tmp_path),
        split_mode="random_frame",
        test_fraction=0.3,
        val_fraction=0.2,
        seed=1,
        read_cube=False,
    )
    dm.setup()
    train, val, test = dm.train_ds, dm.val_ds, dm.test_ds
    assert len(train) and len(val) and len(test)
    assert len(train) + len(val) + len(test) == 20
    assert len(test) == 6  # round(0.3 * 20)
    # test mixes DataSets (not a single-DataSet block)
    assert len({test[i]["dataset"] for i in range(len(test))}) >= 2
    train_stems = {train[i]["stem"] for i in range(len(train))}
    val_stems = {val[i]["stem"] for i in range(len(val))}
    test_stems = {test[i]["stem"] for i in range(len(test))}
    assert train_stems.isdisjoint(val_stems)
    assert (train_stems | val_stems).isdisjoint(test_stems)


def test_getitem_label_only(tmp_path):
    _make_tree(tmp_path, {"DataSet0": 2, "DataSet3": 1})
    dm = MetalScrapDataModule(
        root=str(tmp_path), split_mode="dataset", test_datasets=["DataSet3"], read_cube=False
    )
    dm.setup()
    item = dm.test_ds[0]
    assert "cube" not in item  # read_cube=False
    assert item["label_rgb"].shape == (4, 5, 3)
    assert item["label_rgb"].dtype == np.uint8
    assert item["dataset"] == "DataSet3"
    assert "stem" in item


def test_split_is_deterministic(tmp_path):
    _make_tree(tmp_path, {"DataSet0": 8, "DataSet1": 6, "DataSet3": 2})
    kw = dict(root=str(tmp_path), split_mode="random_frame", val_fraction=0.25, seed=7,
              read_cube=False)
    a = MetalScrapDataModule(**kw)
    a.setup()
    b = MetalScrapDataModule(**kw)
    b.setup()
    a_test = sorted(a.test_ds[i]["stem"] for i in range(len(a.test_ds)))
    b_test = sorted(b.test_ds[i]["stem"] for i in range(len(b.test_ds)))
    a_val = sorted(a.val_ds[i]["stem"] for i in range(len(a.val_ds)))
    b_val = sorted(b.val_ds[i]["stem"] for i in range(len(b.val_ds)))
    assert a_test == b_test
    assert a_val == b_val


def test_validate_params(tmp_path):
    with pytest.raises(ValueError, match="root"):
        MetalScrapDataModule.validate_params({})
    with pytest.raises(ValueError, match="DataSet"):
        (tmp_path / "empty").mkdir()
        MetalScrapDataModule.validate_params({"root": str(tmp_path / "empty")})


def test_real_cube_read(tmp_path):
    """Exercise the cube-reading path against one real frame, if the dataset is present."""
    root = Path("data/HSIMetalScrap")
    if not root.is_dir():
        pytest.skip("real HSIMetalScrap dataset not present")
    dm = MetalScrapDataModule(root=str(root), read_cube=True)
    dm.setup("test")
    item = dm.test_ds[0]
    assert item["cube"].ndim == 3 and item["cube"].shape[-1] == 437
    assert item["label_rgb"].shape[:2] == item["cube"].shape[:2]  # label resized to cube HW
