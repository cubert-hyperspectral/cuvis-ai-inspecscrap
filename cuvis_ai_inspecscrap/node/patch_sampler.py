"""Sample center-pixel patches from labeled pixels (the paper's 7x7-patch / batch-256 protocol).

``mode='train'``: draw ``samples_per_frame`` center pixels per frame (class-balanced by default),
extract a ``patch_size`` x ``patch_size`` window around each (reflect-padded at borders), and emit
``patches [N, P, P, C]`` + integer ``labels [N]``. ``mode='eval'``: take every labeled pixel
(optionally strided to ``max_per_frame``) for dense scoring. ``patch_size=1`` yields single-pixel
spectra for the MLP; ``patch_size=7`` the spatial-spectral patches for the CNNs.

Sampling uses torch's global RNG, so a run that seeds torch is reproducible while still drawing
fresh patches each call.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class PatchSampler(Node):
    """Extract labeled center-pixel patches from a cube + integer target map."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.STOCHASTIC})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1, -1), description="Hyperspectral cube [B,H,W,C]"
        ),
        "targets": PortSpec(
            dtype=torch.int64, shape=(-1, -1, -1), description="Per-pixel class targets [B,H,W]"
        ),
    }
    OUTPUT_SPECS = {
        "patches": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Center-pixel patches [N,P,P,C]",
        ),
        "labels": PortSpec(
            dtype=torch.int64, shape=(-1,), description="Center-pixel class labels [N]"
        ),
    }

    def __init__(
        self,
        patch_size: int = 7,
        samples_per_frame: int = 256,
        class_balanced: bool = True,
        ignore_index: int = -100,
        mode: str = "train",
        max_per_frame: int | None = None,
        **kwargs: Any,
    ) -> None:
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError(f"patch_size must be a positive odd int; got {patch_size}")
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval'; got {mode!r}")
        super().__init__(
            patch_size=patch_size,
            samples_per_frame=samples_per_frame,
            class_balanced=class_balanced,
            ignore_index=ignore_index,
            mode=mode,
            max_per_frame=max_per_frame,
            **kwargs,
        )
        self.patch_size = int(patch_size)
        self.samples_per_frame = int(samples_per_frame)
        self.class_balanced = bool(class_balanced)
        self.ignore_index = int(ignore_index)
        self.mode = mode
        self.max_per_frame = max_per_frame

    def _sample_indices(self, labels: Tensor) -> Tensor:
        """Pick row indices into ``labels`` (class-balanced, with replacement) for training."""
        m = labels.shape[0]
        if not self.class_balanced:
            return torch.randint(0, m, (self.samples_per_frame,))
        present = torch.unique(labels)
        per = max(1, self.samples_per_frame // int(present.numel()))
        picks = []
        for c in present.tolist():
            pos = (labels == c).nonzero(as_tuple=False).reshape(-1)
            picks.append(pos[torch.randint(0, int(pos.numel()), (per,))])
        sel = torch.cat(picks)
        if sel.numel() > self.samples_per_frame:
            sel = sel[torch.randperm(sel.numel())[: self.samples_per_frame]]
        return sel

    @torch.no_grad()
    def forward(self, cube: Tensor, targets: Tensor, **_: Any) -> dict[str, Tensor]:
        """Sample patches + labels across the batch's frames."""
        b, _, _, c = cube.shape
        p = self.patch_size
        r = p // 2
        out_patches: list[Tensor] = []
        out_labels: list[Tensor] = []
        for i in range(b):
            cube_i = cube[i]  # [H,W,C]
            tgt_i = targets[i]  # [H,W]
            coords = (tgt_i != self.ignore_index).nonzero(as_tuple=False)  # [M,2]
            if coords.shape[0] == 0:
                continue
            labels_i = tgt_i[coords[:, 0], coords[:, 1]]  # [M]
            if self.mode == "train":
                sel = self._sample_indices(labels_i)
            else:
                sel = torch.arange(coords.shape[0])
                if self.max_per_frame and sel.numel() > self.max_per_frame:
                    step = math.ceil(sel.numel() / self.max_per_frame)
                    sel = sel[::step]
            ci = coords[sel]  # [N,2]
            lab = labels_i[sel]  # [N]
            if r > 0:
                padded = F.pad(cube_i.permute(2, 0, 1).unsqueeze(0), (r, r, r, r), mode="reflect")[
                    0
                ].permute(1, 2, 0)  # [H+2r, W+2r, C]
            else:
                padded = cube_i
            rows = ci[:, 0].unsqueeze(1) + torch.arange(p, device=ci.device)  # [N,P]
            cols = ci[:, 1].unsqueeze(1) + torch.arange(p, device=ci.device)  # [N,P]
            patches = padded[rows[:, :, None], cols[:, None, :], :]  # [N,P,P,C]
            out_patches.append(patches)
            out_labels.append(lab)

        if not out_patches:
            return {
                "patches": cube.new_zeros((0, p, p, c)),
                "labels": targets.new_zeros((0,), dtype=torch.int64),
            }
        return {"patches": torch.cat(out_patches), "labels": torch.cat(out_labels).to(torch.int64)}


__all__ = ["PatchSampler"]
