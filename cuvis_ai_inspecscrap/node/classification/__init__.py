"""Supervised per-pixel / per-patch HSI classifiers (Gursch et al. 2026)."""

from cuvis_ai_inspecscrap.node.classification.base import PatchClassifierBase
from cuvis_ai_inspecscrap.node.classification.cnn2d import SpatialSpectralCNN2D
from cuvis_ai_inspecscrap.node.classification.cnn3d import SpectralSpatialCNN3D
from cuvis_ai_inspecscrap.node.classification.mlp import SpectralMLPClassifier

__all__ = [
    "PatchClassifierBase",
    "SpectralMLPClassifier",
    "SpatialSpectralCNN2D",
    "SpectralSpatialCNN3D",
]
