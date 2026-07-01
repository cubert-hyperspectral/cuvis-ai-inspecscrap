"""Object-level (per-piece) split patch DataModule, matching the paper's 80/20 protocol.

Gursch et al. split individual scrap objects 80/20 (an object's pixels never cross train/test),
not whole frames. This module reproduces that: objects are the per-frame connected components of the
labeled foreground (the same grouping ``BlobMajorityVote`` uses), and each object is assigned to
train / val / test by a deterministic hash of ``(frame_index, object_id, seed)`` so the split is
reproducible and leak-safe at the piece level. One pass over all frames builds all three splits at
once (each cube is read only once). Patches are served one-per-item, identical to
``MetalScrapPatchDataModule``, so the framework training path is unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch.utils.data import Dataset

from cuvis_ai_inspecscrap.data.metal_scrap_datamodule import MetalScrapDataModule
from cuvis_ai_inspecscrap.data.metal_scrap_patch_datamodule import (
    MetalScrapPatchDataModule,
    _TensorPatchDataset,
)
from cuvis_ai_inspecscrap.node.patch_sampler import PatchSampler

_STAGE_CODE = {"train": 1, "val": 2, "test": 3}


class MetalScrapObjectPatchDataModule(MetalScrapPatchDataModule):
    """Serve cached patches under a piece-level (object) 80/val/20 split."""

    DATA_MODULE_NAME: ClassVar[str] = "metal_scrap_object_patch"

    def _all_frames(self) -> Dataset:
        """Every frame (no frame-level split), cubes read."""
        return MetalScrapDataModule(
            root=str(self.root), split_mode="random_frame", test_fraction=0.0,
            val_fraction=0.0, seed=self.seed, read_cube=True,
        ).build_stage_dataset("train")

    def _obj_stage(self, frame_idx: int, obj_id: int) -> str:
        """Deterministic per-object stage from a hash of (frame, object, seed)."""
        h = (frame_idx * 2654435761 + obj_id * 40503 + self.seed * 2246822519) & 0xFFFFFFFF
        r = h / 0xFFFFFFFF
        if r < self.test_fraction:
            return "test"
        if r < self.test_fraction + 0.10:  # ~10% of objects for val (early-stopping signal)
            return "val"
        return "train"

    def _cache(self, stage: str) -> Path:
        p, s = self.patch_size, self.seed
        if stage in ("train", "val"):
            n = self.train_samples if stage == "train" else self.val_samples
            return self.cache_dir / f"{stage}_ps{p}_seed{s}_objsplit_n{n}.pt"
        return self.cache_dir / f"test_ps{p}_seed{s}_objsplit_cap{self.eval_cap}.pt"

    def _build_all(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """One pass over all frames: connected-component objects -> per-stage patches, all cached."""
        from skimage.measure import label as cc_label

        mapper = self._mapper()
        frame_ds = self._all_frames()
        samplers = {
            "train": PatchSampler(patch_size=self.patch_size, samples_per_frame=self.train_samples,
                                  class_balanced=True, mode="train"),
            "val": PatchSampler(patch_size=self.patch_size, samples_per_frame=self.val_samples,
                                class_balanced=True, mode="train"),
            "test": PatchSampler(patch_size=self.patch_size, mode="eval", max_per_frame=self.eval_cap),
        }
        acc: dict[str, tuple[list, list]] = {s: ([], []) for s in _STAGE_CODE}
        n = len(frame_ds)
        for i in range(n):
            item = frame_ds[i]
            cube = torch.from_numpy(np.ascontiguousarray(item["cube"])).unsqueeze(0).float()
            label = torch.from_numpy(np.ascontiguousarray(item["label_rgb"])).unsqueeze(0)
            tgt = mapper.forward(label_rgb=label)["targets"][0]  # [H,W]
            objmap = cc_label((tgt != -100).cpu().numpy(), connectivity=2)  # 0 = background
            stage_map = np.zeros_like(objmap)
            for o in (int(o) for o in np.unique(objmap) if o != 0):
                stage_map[objmap == o] = _STAGE_CODE[self._obj_stage(i, o)]
            stage_map_t = torch.from_numpy(stage_map)
            for stage, code in _STAGE_CODE.items():
                keep = stage_map_t == code
                if not bool(keep.any()):
                    continue
                masked = torch.where(keep, tgt, torch.full_like(tgt, -100)).unsqueeze(0)
                out = samplers[stage].forward(cube=cube, targets=masked)
                if out["patches"].shape[0]:
                    acc[stage][0].append(out["patches"])
                    acc[stage][1].append(out["labels"])
            if (i + 1) % 20 == 0:
                print(f"[objsplit build] frame {i + 1}/{n}", flush=True)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        c = self.patch_size
        for stage in _STAGE_CODE:
            ps = torch.cat(acc[stage][0]) if acc[stage][0] else torch.zeros((0, c, c, 1))
            ls = torch.cat(acc[stage][1]) if acc[stage][1] else torch.zeros((0,), dtype=torch.int64)
            torch.save({"patches": ps, "labels": ls}, self._cache(stage))
            result[stage] = (ps, ls)
            print(f"[objsplit] {stage}: {ps.shape[0]} patches", flush=True)
        return result

    def build_stage_dataset(self, stage: str) -> Dataset:
        """Load the object-split patches for ``stage`` (building all three on first call)."""
        resolved = "test" if stage == "predict" else stage
        cache_file = self._cache(resolved)
        if cache_file.exists():
            d = torch.load(cache_file)
            return _TensorPatchDataset(d["patches"], d["labels"])
        built = self._build_all()
        ps, ls = built[resolved]
        return _TensorPatchDataset(ps, ls)


__all__ = ["MetalScrapObjectPatchDataModule"]
