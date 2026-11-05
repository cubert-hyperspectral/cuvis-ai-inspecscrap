"""Tests for the Standardize wiring that replaced PerChannelStandardizer.

The bit-identity tests recompute the old formula inline (global per-band population mean/std,
``(x - mean) / (std + eps)``) rather than keeping the removed class, and check it against
``cuvis_ai.node.pretreatments.standardize.Standardize(axis="channel", ddof=0, eps_mode="add")``,
which every pipeline under configs/pipeline/ and reports/models/ now wires directly from the
batch's ``cube`` key (the patch data modules emit ``cube``, not ``patches``; see
cuvis_ai_inspecscrap/data/metal_scrap_patch_datamodule.py and dense_patch_datamodule.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from cuvis_ai.node.pretreatments.standardize import Standardize
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

pytestmark = pytest.mark.unit


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

    standardizer = Standardize(axis="channel", ddof=0, eps=eps, eps_mode="add")
    standardizer.mean.data = old_mean.clone()
    standardizer.std.data = old_std.clone()
    standardizer._statistically_initialized = True
    actual = standardizer.forward(cube=probe)["cube"]

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

    standardizer = Standardize(axis="channel", ddof=0, eps=eps, eps_mode="add")
    standardizer.statistical_initialization(iter([{"cube": fit_data}]))
    actual = standardizer.forward(cube=probe)["cube"]

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
    its standardizer node, fed straight from the batch's ``cube`` key (no bridge node), reproduces
    PerChannelStandardizer's formula given the same stats."""
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
    outputs = pipeline.forward(batch={"cube": probe})
    actual = outputs[("standardizer", "cube")]

    assert torch.equal(actual, expected)


def test_committed_mlp_checkpoint_matches_old_formula_bit_for_bit():
    """Restore the committed ``reports/models/metal_scrap_mlp_framework`` checkpoint through the
    real ``CuvisPipeline.load_pipeline`` weight-restore path (``Standardize._load_from_state_dict``
    resizes the mean/std placeholders to the saved shape, then ``torch.nn.Module`` copies them in;
    this is distinct from hand-assigning ``.mean.data``/``.std.data`` to a freshly constructed node,
    the shortcut every other test in this file takes). Confirms the mean/std actually saved in the
    six committed weight pairs survive that path and still reproduce the removed
    ``PerChannelStandardizer``'s formula bit for bit, backing the migration's bit-identity claim
    against a real checkpoint rather than synthetic stats."""
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "reports" / "models" / "metal_scrap_mlp_framework.yaml"
    weights_path = config_path.with_suffix(".pt")

    pipeline = CuvisPipeline.load_pipeline(str(config_path), weights_path=weights_path)
    nodes = {n.name: n for n in pipeline.nodes}
    standardizer = nodes["standardizer"]
    assert isinstance(standardizer, Standardize)

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    saved_mean = checkpoint["state_dict"]["standardizer"]["mean"]
    saved_std = checkpoint["state_dict"]["standardizer"]["std"]

    torch.manual_seed(4)
    c = saved_mean.numel()
    probe = torch.randn(2, 5, 5, c)
    expected = _old_formula_forward(probe, saved_mean.reshape(c), saved_std.reshape(c), eps=1e-6)

    outputs = pipeline.forward(batch={"cube": probe})
    actual = outputs[("standardizer", "cube")]

    assert torch.equal(actual, expected)
