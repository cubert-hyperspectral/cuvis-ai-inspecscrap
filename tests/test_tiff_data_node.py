"""Test the TiffDataNode batch -> graph adapter (pure tensors, no SDK)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai_inspecscrap.node.data import TiffDataNode

pytestmark = pytest.mark.unit


def test_emits_float32_cube_and_numpy_int32_wavelengths():
    node = TiffDataNode()
    cube = torch.rand((1, 4, 5, 6), dtype=torch.float32)
    wavelengths = torch.tensor(
        [[1000.4, 1200.6, 1400.9, 1600.1, 1000.0, 1300.0]], dtype=torch.float32
    )

    out = node.forward(cube=cube, wavelengths=wavelengths)

    assert out["cube"].dtype == torch.float32
    # Channel selectors require numpy int32 wavelengths [C] (batch dim dropped).
    assert isinstance(out["wavelengths"], np.ndarray)
    assert out["wavelengths"].dtype == np.int32
    assert out["wavelengths"].shape == (6,)
    assert out["wavelengths"].tolist() == [1000, 1200, 1400, 1600, 1000, 1300]
