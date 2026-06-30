"""Tests for PerChannelStandardizer: Phase-1 fit of global per-band mean/std, then standardize."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_inspecscrap.node.normalization import PerChannelStandardizer

pytestmark = pytest.mark.unit


def test_statistical_init_fits_per_channel_stats():
    torch.manual_seed(0)
    c = 5
    # Two batches of patches [B, P, P, C]; known per-channel offset + scale.
    base = torch.randn(40, 3, 3, c)
    offset = torch.tensor([1.0, -2.0, 5.0, 0.0, 10.0])
    scale = torch.tensor([0.5, 2.0, 1.0, 3.0, 0.1])
    data = base * scale + offset
    stream = [{"patches": data[:25]}, {"patches": data[25:]}]

    node = PerChannelStandardizer(in_channels=c)
    node.statistical_initialization(iter(stream))

    flat = data.reshape(-1, c)
    expected_mean = flat.mean(dim=0)
    expected_std = flat.std(dim=0, unbiased=False)
    assert torch.allclose(node.mean, expected_mean, atol=1e-4)
    assert torch.allclose(node.std, expected_std, atol=1e-4)


def test_forward_standardizes_to_zero_mean_unit_std():
    torch.manual_seed(1)
    c = 4
    data = torch.randn(60, 3, 3, c) * torch.tensor([2.0, 0.5, 1.0, 4.0]) + 3.0
    node = PerChannelStandardizer(in_channels=c)
    node.statistical_initialization(iter([{"patches": data}]))

    out = node.forward(patches=data)["normalized"]
    assert out.shape == data.shape
    flat = out.reshape(-1, c)
    assert torch.allclose(flat.mean(dim=0), torch.zeros(c), atol=1e-4)
    assert torch.allclose(flat.std(dim=0, unbiased=False), torch.ones(c), atol=1e-3)


def test_forward_before_fit_raises():
    node = PerChannelStandardizer(in_channels=3)
    with pytest.raises(RuntimeError, match="statistical_initialization"):
        node.forward(patches=torch.randn(2, 3, 3, 3))


def test_requires_initial_fit():
    node = PerChannelStandardizer(in_channels=3)
    assert node.requires_initial_fit is True
