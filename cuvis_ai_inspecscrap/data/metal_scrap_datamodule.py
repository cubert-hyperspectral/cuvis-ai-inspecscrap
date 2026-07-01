"""DataModule for the InSpecScrap HSI metal-scrap dataset (four DataSets, dataset-level split).

The release lays the 170 cubes out as ``<root>/DataSet{0..3}/images/*.tif`` with paired
``<root>/DataSet{0..3}/labels/<stem>.png`` masks. ``tiff_paired`` reads one folder only, so
this module owns its splits (``DataConfig.splits is None``): it enumerates all four DataSets,
tags each sample with its DataSet, and assigns whole DataSets to test (default ``DataSet3``)
while carving a deterministic validation fraction out of the remaining train pool.

A DataSet-level split is a leak-safe proxy for the paper's physical-piece split, valid if the
four DataSets are physically disjoint batches (to confirm with JOANNEUM). Class masks give
within-frame instances (connected components) but not cross-frame piece identity.

Emits per sample ``{"cube" (H,W,C) f32, "wavelengths" int32[C], "label_rgb" (H,W,3) uint8,
"stem", "dataset", "mesu_index"}`` (cube/wavelengths omitted when ``read_cube=False``, e.g. for
the label-sanity baseline).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from torch.utils.data import Dataset

if TYPE_CHECKING:
    # Imported lazily at runtime (see build_stage_dataset) so the plugin registers without the
    # optional [tiff] extra; only pulled in when a TIFF-backed stage is actually built.
    from cuvis_ai_dataloader.data.readers.tiff_reader import TiffCubeReader

_IMAGE_GLOBS = ("tif", "tiff")


def _read_label_rgb(png_path: Path, cube_hw: tuple[int, int] | None) -> np.ndarray:
    """Read a PNG mask as ``(H, W, 3)`` uint8, nearest-resized to ``cube_hw`` when given."""
    from PIL import Image

    img = Image.open(png_path).convert("RGB")
    if cube_hw is not None:
        target_h, target_w = int(cube_hw[0]), int(cube_hw[1])
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8)


class _MetalScrapDataset(Dataset):
    """Torch Dataset over resolved (tiff, png, dataset) samples."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        *,
        reader: TiffCubeReader,
        read_cube: bool,
        label_output_key: str,
    ) -> None:
        self._samples = list(samples)
        self._reader = reader
        self._read_cube = read_cube
        self._label_output_key = label_output_key

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self._samples[idx]
        item: dict[str, Any] = {
            "stem": s["stem"],
            "dataset": s["dataset"],
            "mesu_index": int(idx),
        }
        cube_hw: tuple[int, int] | None = None
        if self._read_cube:
            read = self._reader.read(s["tiff"])
            item["cube"] = read["cube"]
            item["wavelengths"] = read["wavelengths"]
            cube_hw = (read["cube"].shape[0], read["cube"].shape[1])
        item[self._label_output_key] = _read_label_rgb(Path(s["png"]), cube_hw)
        return item


class MetalScrapDataModule(BaseCuvisAIDataModule):
    """Four-DataSet TIFF + paired-PNG DataModule with a dataset-level train/val/test split."""

    DATA_MODULE_NAME: ClassVar[str] = "metal_scrap"

    def __init__(
        self,
        *,
        splits: Any = None,
        batch_size: int = 1,
        num_workers: int = 0,
        root: str | None = None,
        split_mode: str = "random_frame",
        test_fraction: float = 0.2,
        test_datasets: Any = ("DataSet3",),
        val_fraction: float = 0.15,
        seed: int = 42,
        wavelengths: Any = None,
        label_output_key: str = "label_rgb",
        read_cube: bool = True,
        params: dict | None = None,
        data_module: str | None = None,
    ) -> None:
        if params:
            root = root or params.get("root")
            split_mode = params.get("split_mode", split_mode)
            test_fraction = params.get("test_fraction", test_fraction)
            test_datasets = params.get("test_datasets", test_datasets)
            val_fraction = params.get("val_fraction", val_fraction)
            seed = params.get("seed", seed)
            wavelengths = wavelengths if wavelengths is not None else params.get("wavelengths")
            label_output_key = params.get("label_output_key", label_output_key)
            read_cube = params.get("read_cube", read_cube)
        # This module owns its splits; selector splits are not used.
        super().__init__(splits=None, batch_size=batch_size, num_workers=num_workers)
        if split_mode not in ("random_frame", "dataset"):
            raise ValueError(f"split_mode must be 'random_frame' or 'dataset'; got {split_mode!r}")
        self.root = Path(root) if root else None
        self.split_mode = split_mode
        self.test_fraction = float(test_fraction)
        self.test_datasets = (
            [test_datasets] if isinstance(test_datasets, str) else list(test_datasets)
        )
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.wavelengths_override = list(wavelengths) if wavelengths else None
        self.label_output_key = label_output_key
        self.read_cube = bool(read_cube)

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        """Require a ``root`` holding ``DataSet*/images`` directories with TIFFs."""
        root = params.get("root")
        if not root:
            raise ValueError("metal_scrap requires 'root' in params.")
        root_path = Path(root)
        if not root_path.is_dir():
            raise ValueError(f"root does not exist or is not a directory: {root_path}")
        image_dirs = sorted(root_path.glob("DataSet*/images"))
        if not image_dirs:
            raise ValueError(f"no DataSet*/images directories under {root_path}")
        has_tif = any(
            any(d.glob(f"*.{ext}")) for d in image_dirs for ext in _IMAGE_GLOBS
        )
        if not has_tif:
            raise ValueError(f"no *.tif/*.tiff under {root_path}/DataSet*/images")

    def _enumerate_all(self) -> list[dict[str, Any]]:
        """List every (tiff, png, dataset) sample across all DataSets, sorted by source."""
        samples: list[dict[str, Any]] = []
        for images_dir in sorted(self.root.glob("DataSet*/images")):
            dataset = images_dir.parent.name
            labels_dir = images_dir.parent / "labels"
            files: list[Path] = []
            for ext in _IMAGE_GLOBS:
                files.extend(images_dir.glob(f"*.{ext}"))
            for tiff in sorted(set(files)):
                png = labels_dir / f"{tiff.stem}.png"
                if not png.exists():
                    continue
                samples.append(
                    {"tiff": str(tiff), "png": str(png), "stem": tiff.stem, "dataset": dataset}
                )
        samples.sort(key=lambda s: s["tiff"])
        return samples

    def _partition(self) -> dict[str, list[dict[str, Any]]]:
        """Split samples into train/val/test (seeded), then carve a seeded val from the pool.

        ``random_frame`` (default): a seeded ~``test_fraction`` split over all frames, so every
        class appears in train and test (the dataset-level split is distributionally broken here,
        DataSet3 is dark-rust-only). ``dataset``: whole DataSets in ``test_datasets`` form test.
        Both reinstate only a file-level (not piece-level) leakage guard, pending JOANNEUM piece ids.
        """
        all_samples = self._enumerate_all()
        if self.split_mode == "dataset":
            test = [s for s in all_samples if s["dataset"] in self.test_datasets]
            pool = [s for s in all_samples if s["dataset"] not in self.test_datasets]
        else:  # random_frame
            order = list(range(len(all_samples)))
            random.Random(self.seed).shuffle(order)
            n_test = int(round(self.test_fraction * len(all_samples)))
            test_idx = set(order[:n_test])
            test = [all_samples[i] for i in sorted(test_idx)]
            pool = [all_samples[i] for i in range(len(all_samples)) if i not in test_idx]

        pool_sorted = sorted(pool, key=lambda s: s["stem"])
        rng = random.Random(self.seed + 1)
        order = list(range(len(pool_sorted)))
        rng.shuffle(order)
        n_val = int(round(self.val_fraction * len(pool_sorted)))
        val_idx = set(order[:n_val])
        train = [pool_sorted[i] for i in range(len(pool_sorted)) if i not in val_idx]
        val = [pool_sorted[i] for i in range(len(pool_sorted)) if i in val_idx]
        return {"train": train, "val": val, "test": test}

    def build_stage_dataset(self, stage: str) -> Dataset:
        """Build the torch Dataset for one stage (module-owned split)."""
        samples = self._partition().get(stage, [])
        from cuvis_ai_dataloader.data.readers.tiff_reader import TiffCubeReader

        reader = TiffCubeReader(wavelengths_override=self.wavelengths_override)
        return _MetalScrapDataset(
            samples,
            reader=reader,
            read_cube=self.read_cube,
            label_output_key=self.label_output_key,
        )
