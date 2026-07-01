"""Shape / grad / freeze tests for the 2D-CNN and 3D-CNN patch classifiers."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_inspecscrap.node.classification.cnn2d import SpatialSpectralCNN2D
from cuvis_ai_inspecscrap.node.classification.cnn3d import SpectralSpatialCNN3D

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("cls", [SpatialSpectralCNN2D, SpectralSpatialCNN3D])
def test_cnn_forward_shape_and_grad(cls):
    model = cls(in_channels=20, num_classes=5)
    patches = torch.randn(4, 7, 7, 20)  # [N, P, P, C]
    logits = model.forward(patches=patches)["logits"]
    assert logits.shape == (4, 5)
    logits.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("cls", [SpatialSpectralCNN2D, SpectralSpatialCNN3D])
def test_cnn_freeze_unfreeze(cls):
    model = cls(in_channels=12, num_classes=3)
    model.freeze()
    assert all(not p.requires_grad for p in model.parameters())
    model.unfreeze()
    assert all(p.requires_grad for p in model.parameters())


def test_cnn3d_handles_many_bands():
    # 437-band patch should pass through without a shape error.
    model = SpectralSpatialCNN3D(in_channels=437, num_classes=14)
    model.eval()
    logits = model.forward(patches=torch.randn(2, 7, 7, 437))["logits"]
    assert logits.shape == (2, 14)
