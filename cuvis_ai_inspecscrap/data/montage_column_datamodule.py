"""In-memory DataModule that serves one montage column per item.

The metal-scrap notebook builds a side-by-side report (false-RGB | ground truth | prediction | ...)
from images it has already computed (the false-RGB cube, the ground-truth map, the dense prediction,
the cleaned map). To render that report *as a pipeline* rather than a Python loop, each column is one
item of this DataModule: a base RGB image, an integer class map (``None`` for the plain false-RGB
column, served as an all-background map), a caption, and the column position. A
:class:`~cuvis_ai_core.training.predictor.Predictor` then streams the columns through
``ClassMapToRGB -> LabelOverlay -> TitleOverlay -> MontageColumnSink``.

Unlike the disk-backed modules, this one holds its columns in memory (they are derived tensors, not
files), so it is constructed directly with ``columns=`` rather than from a manifest path.

Emits per item ``{"rgb_image" (H,W,3) f32, "class_map" (H,W) int64, "caption" str,
"column_index" int}``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from torch import Tensor
from torch.utils.data import Dataset

#: One montage column: ``(caption, base_rgb [H,W,3], class_map [H,W] | None)``.
Column = tuple[str, Tensor, Tensor | None]


class _MontageColumnDataset(Dataset):
    """Wrap pre-computed montage columns; a ``None`` class map becomes an all-background map."""

    def __init__(self, columns: list[Column], background_value: int = -1) -> None:
        self._columns = list(columns)
        self.background_value = int(background_value)

    def __len__(self) -> int:
        return len(self._columns)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        caption, rgb, class_map = self._columns[idx]
        rgb_t = torch.as_tensor(rgb, dtype=torch.float32)
        if class_map is None:
            class_map_t = torch.full(rgb_t.shape[:2], self.background_value, dtype=torch.long)
        else:
            class_map_t = torch.as_tensor(class_map, dtype=torch.long)
        return {
            "rgb_image": rgb_t,
            "class_map": class_map_t,
            "caption": str(caption),
            "column_index": idx,
        }


class MontageColumnDataModule(BaseCuvisAIDataModule):
    """Serve pre-computed montage columns one per item (batch = all columns, in order)."""

    DATA_MODULE_NAME: ClassVar[str] = "metal_scrap_montage_columns"

    def __init__(
        self,
        *,
        columns: list[Column] | None = None,
        background_value: int = -1,
        batch_size: int | None = None,
        num_workers: int = 0,
        splits: Any = None,
        params: dict | None = None,
        data_module: str | None = None,
    ) -> None:
        if params:
            columns = columns if columns is not None else params.get("columns")
            background_value = params.get("background_value", background_value)
        cols: list[Column] = list(columns) if columns is not None else []
        # One batch holds every column by default, so the sink sees them all in a single forward.
        super().__init__(
            splits=None, batch_size=batch_size or max(1, len(cols)), num_workers=num_workers
        )
        self._columns = cols
        self.background_value = int(background_value)

    def build_stage_dataset(self, stage: str) -> Dataset:
        """Every stage serves the same in-memory columns (rendering has no train/test split)."""
        return _MontageColumnDataset(self._columns, self.background_value)


__all__ = ["MontageColumnDataModule", "_MontageColumnDataset", "Column"]
