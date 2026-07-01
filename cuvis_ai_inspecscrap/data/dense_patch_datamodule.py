"""Dense, lazy patch DataModule for tiling-style inference over whole frames.

``MetalScrapPatchDataModule`` materializes a *sample* of patches in memory for training. Dense
inference instead needs **every** labelled pixel of a frame, which is far too many patches to
stack at once (a full frame is multi-GB of 7x7x437 windows). This module serves one patch per item
and **gathers each 7x7 window lazily in ``__getitem__``**, so a ``DataLoader`` with
``batch_size=CHUNK`` keeps peak memory at one batch, not the whole frame. ``batch_size`` IS the
inference chunk.

Every item carries its provenance ``(frame_id, y, x)`` plus the source ``height``/``width`` so a
downstream sink (``ClassMapAccumulator``) can scatter predictions back into ``[H, W]`` class maps.
Only the reflect-padded cubes of the selected frames stay resident; patches are never pre-expanded.

Emits per item ``{"patches" (P,P,C) f32, "frame_id" int, "y" int, "x" int, "height" int,
"width" int, "targets" int}``. ``targets`` is the center pixel's class label, so a metrics sink
(``MulticlassSegmentationMetrics``) can score the dense pass during inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from torch import Tensor
from torch.utils.data import Dataset

from cuvis_ai_inspecscrap.data.metal_scrap_datamodule import MetalScrapDataModule
from cuvis_ai_inspecscrap.node.labels import RgbLabelToClassIndex


class _DensePatchDataset(Dataset):
    """Lazy dense patches over a frame set, tagged with ``(frame_id, y, x)`` for later rebuild.

    The reflect-padded cubes stay resident (one per selected frame); ``__getitem__`` slices a single
    ``P x P`` window on demand, so memory is ``O(frames + one batch)``, never ``O(all patches)``.
    """

    def __init__(self, frames: list[dict[str, Any]], patch_size: int, ignore_index: int) -> None:
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError(f"patch_size must be a positive odd int; got {patch_size}")
        self.patch_size = int(patch_size)
        radius = self.patch_size // 2
        self._padded: list[Tensor] = []
        self._targets: list[Tensor] = []
        self.frame_meta: list[dict[str, Any]] = []
        item_rows: list[Tensor] = []
        for frame_id, frame in enumerate(frames):
            cube = torch.from_numpy(np.ascontiguousarray(frame["cube"])).float()  # [H,W,C]
            padded = F.pad(
                cube.permute(2, 0, 1).unsqueeze(0), (radius, radius, radius, radius), mode="reflect"
            )[0].permute(1, 2, 0).contiguous()  # [H+2r, W+2r, C]
            self._padded.append(padded)
            targets = frame["targets"]  # [H,W] int64
            self._targets.append(torch.as_tensor(targets, dtype=torch.long))
            height, width = int(targets.shape[0]), int(targets.shape[1])
            self.frame_meta.append({"stem": frame.get("stem", str(frame_id)), "height": height, "width": width})
            coords = (targets != ignore_index).nonzero(as_tuple=False)  # [M,2] row-major
            fid_col = torch.full((coords.shape[0], 1), frame_id, dtype=torch.long)
            item_rows.append(torch.cat([fid_col, coords.to(torch.long)], dim=1))  # [M,3]
        self._items = (
            torch.cat(item_rows) if item_rows else torch.zeros((0, 3), dtype=torch.long)
        )  # [N,3] = (frame_id, y, x)

    def __len__(self) -> int:
        return int(self._items.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        frame_id, y, x = (int(v) for v in self._items[idx].tolist())
        p = self.patch_size
        patch = self._padded[frame_id][y : y + p, x : x + p, :]  # center is original (y, x)
        meta = self.frame_meta[frame_id]
        return {
            "patches": patch,
            "frame_id": frame_id,
            "y": y,
            "x": x,
            "height": meta["height"],
            "width": meta["width"],
            "targets": int(self._targets[frame_id][y, x]),  # center label (for the metrics sink)
        }


class DensePatchDataModule(BaseCuvisAIDataModule):
    """Serve every labelled pixel of selected frames as lazy ``batch_size``-wide patch batches."""

    DATA_MODULE_NAME: ClassVar[str] = "metal_scrap_dense_patch"

    def __init__(
        self,
        *,
        splits: Any = None,
        batch_size: int = 1024,
        num_workers: int = 0,
        root: str | None = None,
        patch_size: int = 7,
        frame_indices: Any = None,
        ignore_index: int = -100,
        split_mode: str = "random_frame",
        test_fraction: float = 0.2,
        val_fraction: float = 0.15,
        seed: int = 42,
        labelmap_path: str | None = None,
        params: dict | None = None,
        data_module: str | None = None,
    ) -> None:
        if params:
            root = root or params.get("root")
            patch_size = params.get("patch_size", patch_size)
            frame_indices = frame_indices if frame_indices is not None else params.get("frame_indices")
            ignore_index = params.get("ignore_index", ignore_index)
            split_mode = params.get("split_mode", split_mode)
            test_fraction = params.get("test_fraction", test_fraction)
            val_fraction = params.get("val_fraction", val_fraction)
            seed = params.get("seed", seed)
            labelmap_path = labelmap_path or params.get("labelmap_path")
        super().__init__(splits=None, batch_size=batch_size, num_workers=num_workers)
        self.root = Path(root) if root else None
        self.patch_size = int(patch_size)
        self.frame_indices = None if frame_indices is None else [int(i) for i in frame_indices]
        self.ignore_index = int(ignore_index)
        self.split_mode = split_mode
        self.test_fraction = float(test_fraction)
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.labelmap_path = (
            str(labelmap_path)
            if labelmap_path
            else (str(self.root / "LabelMap.txt") if self.root else None)
        )

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        """Require a ``root`` holding ``DataSet*/images`` directories (delegated)."""
        MetalScrapDataModule.validate_params(params)

    # -- overridable seam (tests inject a synthetic frame source) --------------
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
        if self.labelmap_path is None:
            raise ValueError("DensePatchDataModule needs a labelmap_path or root.")
        return RgbLabelToClassIndex(labelmap_path=self.labelmap_path)

    def _frames(self, stage: str) -> list[dict[str, Any]]:
        """Read the selected frames once and map their PNG masks to integer class targets."""
        frame_ds = self._frame_dataset(stage)
        mapper = self._mapper()
        indices = self.frame_indices if self.frame_indices is not None else range(len(frame_ds))
        frames: list[dict[str, Any]] = []
        for idx in indices:
            item = frame_ds[idx]
            label = torch.from_numpy(np.ascontiguousarray(item["label_rgb"])).unsqueeze(0)
            targets = mapper.forward(label_rgb=label)["targets"][0]
            frames.append({"cube": item["cube"], "targets": targets, "stem": item["stem"]})
        return frames

    def build_stage_dataset(self, stage: str) -> Dataset:
        """Build the dense patch dataset (``predict`` densifies the test frames)."""
        resolved = "test" if stage == "predict" else stage
        return _DensePatchDataset(self._frames(resolved), self.patch_size, self.ignore_index)


__all__ = ["DensePatchDataModule", "_DensePatchDataset"]
