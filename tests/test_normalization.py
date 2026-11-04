"""Tests for PatchesToCube and the Standardize wiring that replaced PerChannelStandardizer.

The bit-identity test recomputes the old formula inline (global per-band population mean/std,
``(x - mean) / (std + eps)``) rather than keeping the removed class, and checks it against
``cuvis_ai.node.pretreatments.standardize.Standardize(axis="channel", ddof=0, eps_mode="add")``
fed through ``PatchesToCube``, which is the wiring every pipeline under configs/pipeline/ and
reports/models/ now uses.
"""

from __future__ import annotations

import pytest
import torch
from cuvis_ai.node.pretreatments.standardize import Standardize

from cuvis_ai_inspecscrap.node.normalization import PatchesToCube

pytestmark = pytest.mark.unit


def test_patches_to_cube_renames_port_without_touching_values():
    node = PatchesToCube()
    patches = torch.randn(6, 7, 7, 5)
    out = node.forward(patches=patches)
    assert set(out) == {"cube"}
    assert torch.equal(out["cube"], patches)


def _old_formula_fit(fit_data: torch.Tensor):
    """Global per-band population mean/std, the removed ``PerChannelStandardizer`` fit
    (``total``/``total_sq`` accumulation, recomputed here, not kept as a class)."""
    c = fit_data.shape[-1]
    flat = fit_data.reshape(-1, c).to(torch.float64)
    total = flat.sum(dim=0)
    total_sq = (flat * flat).sum(dim=0)
    count = flat.shape[0]
    mean = total / count
    var = (total_sq / count) - mean * mean
    std = var.clamp_min(0.0).sqrt()
    return mean.to(torch.float32), std.to(torch.float32)


def _old_formula_forward(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float):
    """``(x - mean) / (std + eps)`` along the channel axis, the removed class's ``forward``."""
    return (data - mean) / (std + eps)


def test_standardize_forward_matches_old_formula_bit_for_bit_given_the_same_stats():
    """The forward formula ``PerChannelStandardizer`` -> ``Standardize`` is a straight port: given
    the same mean/std (as a loaded checkpoint carries), the two compute the identical arithmetic.
    This is what the committed reports/models/*.pt weights rely on across the migration."""
    torch.manual_seed(1)
    c = 5
    eps = 1e-6
    fit_data = torch.randn(40, 3, 3, c) * torch.tensor([0.5, 2.0, 1.0, 3.0, 0.1]) + torch.tensor(
        [1.0, -2.0, 5.0, 0.0, 10.0]
    )
    probe = torch.randn(8, 3, 3, c)
    old_mean, old_std = _old_formula_fit(fit_data)
    expected = _old_formula_forward(probe, old_mean, old_std, eps)

    bridge = PatchesToCube()
    standardizer = Standardize(axis="channel", ddof=0, eps=eps, eps_mode="add")
    standardizer.mean.data = old_mean.clone()
    standardizer.std.data = old_std.clone()
    standardizer._statistically_initialized = True
    actual = standardizer.forward(cube=bridge.forward(patches=probe)["cube"])["cube"]

    assert torch.equal(actual, expected)


def test_standardize_fit_matches_old_formula_closely():
    """``Standardize``'s fit (Chan's parallel formula, torch.var_mean) and the removed class's fit
    (raw sum / sum-of-squares) are different, numerically-equivalent float64 algorithms for the
    same population mean/std; they agree to float32 precision, not bit for bit."""
    torch.manual_seed(2)
    c = 5
    eps = 1e-6
    fit_data = torch.randn(40, 3, 3, c) * torch.tensor([0.5, 2.0, 1.0, 3.0, 0.1]) + torch.tensor(
        [1.0, -2.0, 5.0, 0.0, 10.0]
    )
    probe = torch.randn(8, 3, 3, c)
    old_mean, old_std = _old_formula_fit(fit_data)
    expected = _old_formula_forward(probe, old_mean, old_std, eps)

    bridge = PatchesToCube()
    standardizer = Standardize(axis="channel", ddof=0, eps=eps, eps_mode="add")
    standardizer.statistical_initialization(
        iter([{"cube": bridge.forward(patches=fit_data)["cube"]}])
    )
    actual = standardizer.forward(cube=bridge.forward(patches=probe)["cube"])["cube"]

    assert torch.allclose(actual, expected, atol=1e-4)


def test_standardize_requires_fit_before_forward():
    standardizer = Standardize(axis="channel", ddof=0, eps=1e-6, eps_mode="add")
    with pytest.raises(RuntimeError, match="statistical_initialization"):
        standardizer.forward(cube=torch.randn(2, 3, 3, 3))


def test_standardize_requires_initial_fit():
    standardizer = Standardize(axis="channel", ddof=0, eps=1e-6, eps_mode="add")
    assert standardizer.requires_initial_fit is True


def test_migrated_pipeline_standardizer_matches_old_formula_bit_for_bit():
    """Load configs/pipeline/metal_scrap_mlp.yaml exactly as restore-pipeline would and confirm
    its standardizer node reproduces PerChannelStandardizer's formula given the same stats."""
    from pathlib import Path

    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

    repo_root = Path(__file__).resolve().parents[1]
    pipeline = CuvisPipeline.load_pipeline(
        str(repo_root / "configs" / "pipeline" / "metal_scrap_mlp.yaml")
    )
    nodes = {n.name: n for n in pipeline.nodes}
    standardizer = nodes["standardizer"]
    assert isinstance(standardizer, Standardize)

    torch.manual_seed(3)
    c = 437
    probe = torch.randn(2, 7, 7, c)
    old_mean = torch.randn(c) * 2.0 + 1.0
    old_std = torch.rand(c) * 3.0 + 0.1
    standardizer.mean.data = old_mean.view(1, 1, 1, c).clone()
    standardizer.std.data = old_std.view(1, 1, 1, c).clone()
    standardizer._statistically_initialized = True

    expected = _old_formula_forward(probe, old_mean, old_std, eps=1e-6)
    outputs = pipeline.forward(batch={"patches": probe})
    actual = outputs[("standardizer", "cube")]

    assert torch.equal(actual, expected)
