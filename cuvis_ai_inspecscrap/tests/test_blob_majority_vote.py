"""Tests for BlobMajorityVote: supplied-instances and derived-from-foreground voting."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_inspecscrap.node.deciders import BlobMajorityVote

pytestmark = pytest.mark.unit


def test_supplied_instances_majority_vote():
    preds = torch.tensor([[[0, 0, 1, 2], [0, 1, 1, 1]]], dtype=torch.int64)  # [1,2,4]
    instances = torch.tensor([[[1, 1, 2, 2], [1, 1, 2, 2]]], dtype=torch.int64)
    # object 1 (left half): preds {0,0,0,1} -> 0 ; object 2 (right half): {1,2,1,1} -> 1
    out = BlobMajorityVote(num_classes=3).forward(predictions=preds, instances=instances)
    assert out["predictions"][0].tolist() == [[0, 0, 1, 1], [0, 0, 1, 1]]


def test_ignore_instance_left_unchanged():
    preds = torch.tensor([[[5, 0], [0, 0]]], dtype=torch.int64)
    instances = torch.tensor([[[0, 1], [1, 1]]], dtype=torch.int64)  # id 0 = background (ignored)
    out = BlobMajorityVote(num_classes=6, ignore_instance=0).forward(
        predictions=preds, instances=instances
    )
    # object 1 -> majority 0; the ignored (0,0) pixel keeps its original 5
    assert out["predictions"][0].tolist() == [[5, 0], [0, 0]]


def test_derive_instances_from_foreground():
    preds = torch.tensor([[[0, 1, 9, 2, 3]]], dtype=torch.int64)  # [1,1,5]
    foreground = torch.tensor([[[1, 1, 0, 1, 1]]], dtype=torch.int64)  # two blobs split by the gap
    out = BlobMajorityVote(num_classes=10).forward(predictions=preds, foreground=foreground)
    # blob A {0,1} -> tie -> argmax 0 ; gap pixel keeps 9 ; blob B {2,3} -> tie -> argmax 2
    assert out["predictions"][0].tolist() == [[0, 0, 9, 2, 2]]


def test_requires_instances_or_foreground():
    with pytest.raises(ValueError, match="instances.*foreground"):
        BlobMajorityVote().forward(predictions=torch.zeros(1, 2, 2, dtype=torch.int64))
