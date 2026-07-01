"""Patch-serving DataModule for the framework training path (Gursch et al. 2026 protocol).

``MetalScrapDataModule`` yields whole frames; the paper trains on fixed-size center-pixel patches
in batches of 256. This module turns frames into patches once (the ``PatchSampler`` node, reusing an
on-disk cache so the cube IO is paid only the first time per ``patch_size``) and serves **one patch
per item**, so a ``DataLoader`` with ``batch_size=256`` collates to the paper's ``[256, P, P, C]`` +
``[256]`` batch. The pipeline that consumes this is just ``classifier -> loss + metrics``: the batch
keys ``patches`` / ``targets`` auto-inject into those nodes' input ports.

Train / val draw a class-balanced sample per frame (``train_samples`` / ``val_samples``); test takes
a strided dense sample (``eval_cap`` per frame, optionally bounding the number of test frames) so the
7x7-patch memory stays in check. The frame split, label colour->class merge, and cache file naming
match ``MetalScrapDataModule`` + the standalone caching path, so the existing ``reports/cache`` files
are reused as-is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from torch import Tensor
from torch.utils.data import Dataset

from cuvis_ai_inspecscrap.data.metal_scrap_datamodule import MetalScrapDataModule
from cuvis_ai_inspecscrap.node.labels import RgbLabelToClassIndex
from cuvis_ai_inspecscrap.node.patch_sampler import PatchSampler


class _TensorPatchDataset(Dataset):
    """In-memory patch pool: one ``{"patches": [P,P,C], "targets": int}`` item per patch."""

    def __init__(self, patches: Tensor, labels: Tensor) -> None:
        self.patches = patches
        self.labels = labels.to(torch.int64)

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"patches": self.patches[idx], "targets": int(self.labels[idx])}


class MetalScrapPatchDataModule(BaseCuvisAIDataModule):
    """Serve cached center-pixel patches as ``batch_size``-wide ``patches`` + ``targets`` batches."""

    DATA_MODULE_NAME: ClassVar[str] = "metal_scrap_patch"

    def __init__(
        self,
        *,
        splits: Any = None,
        batch_size: int = 256,
        num_workers: int = 0,
        root: str | None = None,
        patch_size: int = 7,
        train_samples: int = 600,
        val_samples: int = 400,
        eval_cap: int = 5000,
        max_train_frames: int = 0,
        max_val_frames: int = 0,
        max_test_frames: int = 0,
        seed: int = 42,
        split_mode: str = "random_frame",
        test_fraction: float = 0.2,
        val_fraction: float = 0.15,
        cache_dir: str | None = None,
        labelmap_path: str | None = None,
    ) -> None:
        # This module owns its splits (frames are partitioned by MetalScrapDataModule).
        super().__init__(splits=None, batch_size=batch_size, num_workers=num_workers)
        self.root = Path(root) if root else None
        self.patch_size = int(patch_size)
        self.train_samples = int(train_samples)
        self.val_samples = int(val_samples)
        self.eval_cap = int(eval_cap)
        self.max_train_frames = int(max_train_frames)
        self.max_val_frames = int(max_val_frames)
        self.max_test_frames = int(max_test_frames)
        self.seed = int(seed)
        self.split_mode = split_mode
        self.test_fraction = float(test_fraction)
        self.val_fraction = float(val_fraction)
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else (Path(__file__).resolve().parents[2] / ".patch_cache")
        )
        self.labelmap_path = (
            str(labelmap_path)
            if labelmap_path
            else (str(self.root / "LabelMap.txt") if self.root else None)
        )

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        """Require a ``root`` holding ``DataSet*/images`` directories (delegated)."""
        MetalScrapDataModule.validate_params(params)

    # -- overridable seams (tests inject a synthetic frame source / mapper) ----
    def _frame_dataset(self, stage: str) -> Dataset:
        """Whole-frame dataset for ``stage`` (``{"cube", "label_rgb", "stem"}`` per item)."""
        frame_dm = MetalScrapDataModule(
            root=str(self.root),
            split_mode=self.split_mode,
            test_fraction=self.test_fraction,
            val_fraction=self.val_fraction,
            seed=self.seed,
            read_cube=True,
        )
        return frame_dm.build_stage_dataset(stage)

    def _mapper(self) -> RgbLabelToClassIndex:
        """The colour->class-index label mapper (provisional 29->14 merge)."""
        if self.labelmap_path is None:
            raise ValueError("MetalScrapPatchDataModule needs a labelmap_path or root.")
        return RgbLabelToClassIndex(labelmap_path=self.labelmap_path)

    # -- patch extraction ------------------------------------------------------
    def _accumulate(
        self, frame_ds: Dataset, mapper: RgbLabelToClassIndex, sampler: PatchSampler, max_frames: int
    ) -> tuple[Tensor, Tensor]:
        """Read frames once, sample patches, concatenate to ``([N,P,P,C], [N])`` on CPU."""
        patches: list[Tensor] = []
        labels: list[Tensor] = []
        n = len(frame_ds) if not max_frames else min(max_frames, len(frame_ds))
        for i in range(n):
            item = frame_ds[i]
            cube = torch.from_numpy(np.ascontiguousarray(item["cube"])).unsqueeze(0).float()
            label = torch.from_numpy(np.ascontiguousarray(item["label_rgb"])).unsqueeze(0)
            targets = mapper.forward(label_rgb=label)["targets"]
            out = sampler.forward(cube=cube, targets=targets)
            if out["patches"].shape[0]:
                patches.append(out["patches"])
                labels.append(out["labels"])
        if not patches:
            c = int(self.patch_size)
            return torch.zeros((0, c, c, 1)), torch.zeros((0,), dtype=torch.int64)
        return torch.cat(patches), torch.cat(labels)

    def _cache_file(self, stage: str) -> Path:
        """Cache filename, matching the standalone caching path for train/val reuse."""
        p, s = self.patch_size, self.seed
        if stage in ("train", "val"):
            n = self.train_samples if stage == "train" else self.val_samples
            return self.cache_dir / f"{stage}_ps{p}_seed{s}_n{n}.pt"
        suffix = f"_mtf{self.max_test_frames}" if self.max_test_frames else ""
        return self.cache_dir / f"test_ps{p}_seed{s}_cap{self.eval_cap}{suffix}.pt"

    def _load_or_build(self, stage: str) -> tuple[Tensor, Tensor]:
        """Return cached patches for ``stage`` or build, cache, and return them."""
        cache_file = self._cache_file(stage)
        if cache_file.exists():
            d = torch.load(cache_file)
            return d["patches"], d["labels"]

        mapper = self._mapper()
        frame_ds = self._frame_dataset(stage)
        if stage in ("train", "val"):
            n = self.train_samples if stage == "train" else self.val_samples
            max_frames = self.max_train_frames if stage == "train" else self.max_val_frames
            sampler = PatchSampler(
                patch_size=self.patch_size, samples_per_frame=n, class_balanced=True, mode="train"
            )
        else:
            max_frames = self.max_test_frames
            sampler = PatchSampler(
                patch_size=self.patch_size, mode="eval", max_per_frame=self.eval_cap
            )
        patches, labels = self._accumulate(frame_ds, mapper, sampler, max_frames)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"patches": patches, "labels": labels}, cache_file)
        return patches, labels

    def build_stage_dataset(self, stage: str) -> Dataset:
        """Build the per-stage patch dataset (``predict`` reuses the ``test`` patches)."""
        resolved = "test" if stage == "predict" else stage
        patches, labels = self._load_or_build(resolved)
        return _TensorPatchDataset(patches, labels)


__all__ = ["MetalScrapPatchDataModule", "_TensorPatchDataset"]
