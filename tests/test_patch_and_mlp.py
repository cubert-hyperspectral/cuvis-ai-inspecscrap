"""Tests for PatchSampler (sampling + center extraction) and SpectralMLPClassifier."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_inspecscrap.node.classification.mlp import SpectralMLPClassifier
from cuvis_ai_inspecscrap.node.patch_sampler import PatchSampler

pytestmark = pytest.mark.unit


def _labeled_cube():
    """A 4x4 frame whose channel 0 equals the pixel's class id (so center==label is checkable)."""
    targets = torch.tensor(
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=torch.int64
    ).unsqueeze(0)  # [1,4,4]
    cube = torch.zeros(1, 4, 4, 2, dtype=torch.float32)
    cube[0, :, :, 0] = targets[0].float()  # channel 0 = class id
    cube[0, :, :, 1] = 7.0  # constant marker channel
    return cube, targets


def test_patch_sampler_size1_center_matches_label():
    cube, targets = _labeled_cube()
    sampler = PatchSampler(patch_size=1, samples_per_frame=8, class_balanced=True, mode="train")
    out = sampler.forward(cube=cube, targets=targets)
    patches, labels = out["patches"], out["labels"]
    assert patches.shape[1:] == (1, 1, 2)
    assert labels.shape[0] == patches.shape[0] > 0
    # center channel-0 encodes the class id -> must equal the label
    assert torch.equal(patches[:, 0, 0, 0].to(torch.int64), labels)
    assert torch.all(patches[:, 0, 0, 1] == 7.0)
    # class-balanced sampling reaches every present class
    assert set(labels.tolist()) == {0, 1, 2, 3}


def test_patch_sampler_size3_reflect_and_center():
    cube, targets = _labeled_cube()
    sampler = PatchSampler(patch_size=3, samples_per_frame=12, mode="train")
    out = sampler.forward(cube=cube, targets=targets)
    patches, labels = out["patches"], out["labels"]
    assert patches.shape[1:] == (3, 3, 2)  # reflect-padded at the borders
    assert torch.equal(patches[:, 1, 1, 0].to(torch.int64), labels)  # patch center == label


def test_patch_sampler_eval_dense_and_cap():
    cube, targets = _labeled_cube()
    dense = PatchSampler(patch_size=1, mode="eval").forward(cube=cube, targets=targets)
    assert dense["patches"].shape[0] == 16  # every labeled pixel
    capped = PatchSampler(patch_size=1, mode="eval", max_per_frame=4).forward(
        cube=cube, targets=targets
    )
    assert 0 < capped["patches"].shape[0] <= 16


def test_patch_sampler_skips_ignored_centers():
    targets = torch.full((1, 4, 4), -100, dtype=torch.int64)
    targets[0, 0, 0] = 1  # single labeled pixel
    cube = torch.zeros(1, 4, 4, 2)
    out = PatchSampler(patch_size=1, mode="eval").forward(cube=cube, targets=targets)
    assert out["patches"].shape[0] == 1
    assert out["labels"].tolist() == [1]


def test_mlp_forward_shape_and_grad():
    mlp = SpectralMLPClassifier(in_channels=5, num_classes=3)
    patches = torch.randn(10, 1, 1, 5)
    logits = mlp.forward(patches=patches)["logits"]
    assert logits.shape == (10, 3)
    logits.sum().backward()
    assert any(p.grad is not None for p in mlp.parameters())


def test_mlp_uses_patch_center_for_p7():
    mlp = SpectralMLPClassifier(in_channels=4, num_classes=2)
    patches = torch.randn(6, 7, 7, 4)
    assert mlp.forward(patches=patches)["logits"].shape == (6, 2)


def test_mlp_freeze_unfreeze():
    mlp = SpectralMLPClassifier(in_channels=4, num_classes=2)
    mlp.freeze()
    assert all(not p.requires_grad for p in mlp.parameters())
    mlp.unfreeze()
    assert all(p.requires_grad for p in mlp.parameters())
