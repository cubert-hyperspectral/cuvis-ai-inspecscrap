"""Tests for WeightedCrossEntropyLoss and MulticlassSegmentationMetrics."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from cuvis_ai_inspecscrap.node.losses import (
    InverseFrequencyClassWeights,
    WeightedCrossEntropyLoss,
    inverse_frequency_from_counts,
)
from cuvis_ai_inspecscrap.node.metrics import MulticlassSegmentationMetrics

pytestmark = pytest.mark.unit


def _confident_logits(targets: torch.Tensor, k: int) -> torch.Tensor:
    """One-hot-ish logits that argmax to ``targets`` (near-zero CE)."""
    logits = torch.full((targets.numel(), k), -10.0)
    logits[torch.arange(targets.numel()), targets] = 10.0
    return logits


# --- WeightedCrossEntropyLoss -------------------------------------------------


def test_loss_perfect_prediction_near_zero():
    node = WeightedCrossEntropyLoss(num_classes=3)
    targets = torch.tensor([0, 1, 2, 0])
    loss = node.forward(logits=_confident_logits(targets, 3), targets=targets)["loss"]
    assert loss.ndim == 0
    assert loss.item() < 1e-3


def test_loss_weight_scales_linearly():
    targets = torch.tensor([0, 1, 2])
    logits = torch.randn(3, 3)
    a = WeightedCrossEntropyLoss(num_classes=3, weight=1.0).forward(logits=logits, targets=targets)
    b = WeightedCrossEntropyLoss(num_classes=3, weight=2.0).forward(logits=logits, targets=targets)
    assert torch.allclose(b["loss"], 2.0 * a["loss"], rtol=1e-5)


def test_loss_ignore_index_and_grad():
    node = WeightedCrossEntropyLoss(num_classes=3, ignore_index=-100)
    logits = torch.randn(3, 3, requires_grad=True)
    targets = torch.tensor([0, 1, -100])  # last is ignored
    loss = node.forward(logits=logits, targets=targets)["loss"]
    loss.backward()
    assert logits.grad is not None
    # Gradient only flows through the two non-ignored rows.
    assert torch.count_nonzero(logits.grad.abs().sum(dim=1)) == 2


def test_loss_all_ignored_stays_connected():
    node = WeightedCrossEntropyLoss(num_classes=3)
    logits = torch.randn(2, 3, requires_grad=True)
    loss = node.forward(logits=logits, targets=torch.tensor([-100, -100]))["loss"]
    assert loss.item() == 0.0
    loss.backward()  # connected to the graph, no error


def test_loss_class_weights_validated():
    with pytest.raises(ValueError, match="class_weights"):
        WeightedCrossEntropyLoss(num_classes=3, class_weights=[1.0, 1.0])


def test_inverse_frequency_weights_golden():
    # labels [0,0,0,1], k=2 -> counts [3,1], total 4 -> w0 = 4/(2*3), w1 = 4/(2*1).
    w = WeightedCrossEntropyLoss.inverse_frequency_weights(torch.tensor([0, 0, 0, 1]), 2)
    assert w == pytest.approx([2 / 3, 2.0])


def test_inverse_frequency_weights_absent_class_is_zero():
    # class 2 never appears -> weight 0.0; present classes share the inverse-frequency mass.
    w = WeightedCrossEntropyLoss.inverse_frequency_weights(torch.tensor([0, 0, 1, 1]), 3)
    assert w[2] == 0.0
    assert w[0] == pytest.approx(4 / (3 * 2)) and w[1] == pytest.approx(4 / (3 * 2))


def test_set_class_weights_applies_and_validates():
    node = WeightedCrossEntropyLoss(num_classes=4)  # constructed without weights
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    targets = torch.tensor([0, 3])
    unweighted = node.forward(logits=logits, targets=targets)["loss"].item()
    node.set_class_weights([1.0, 2.0, 3.0, 4.0])
    weighted = node.forward(logits=logits, targets=targets)["loss"].item()
    assert weighted != pytest.approx(unweighted)  # the weighting now takes effect
    assert "_class_weights" in dict(node.named_buffers())  # a registered buffer, not a plain attr
    with pytest.raises(ValueError, match="class_weights"):
        node.set_class_weights([1.0, 2.0])  # wrong length


def test_fit_class_weights_matches_static_and_serializes():
    labels = torch.tensor([0, 0, 0, 1, 2, 2])
    node = WeightedCrossEntropyLoss(num_classes=3)
    node.fit_class_weights(labels)
    expected = WeightedCrossEntropyLoss.inverse_frequency_weights(labels, 3)
    assert node._class_weights.tolist() == pytest.approx(expected)
    # the derived weights round-trip through state_dict (buffer travels with the node)
    fresh = WeightedCrossEntropyLoss(num_classes=3, class_weights=[1.0, 1.0, 1.0])
    fresh.load_state_dict(node.state_dict())
    assert fresh._class_weights.tolist() == pytest.approx(expected)


# --- InverseFrequencyClassWeights + class-weight plumbing --------------------


def test_inverse_frequency_from_counts_golden():
    # counts [3, 1, 2, 0], total 6, num_classes 4 -> 6/(4*3), 6/(4*1), 6/(4*2); zero count -> 0.0.
    w = inverse_frequency_from_counts(torch.tensor([3, 1, 2, 0]), 4)
    assert w.tolist() == pytest.approx([0.5, 1.5, 0.75, 0.0])


def test_class_weights_estimator_fits_from_stream():
    # Two target batches; fitted weights match the static formula on the concatenation.
    labels = torch.tensor([0, 0, 0, 1, 2, 2])
    stream = [{"targets": labels[:4]}, {"targets": labels[4:]}]
    node = InverseFrequencyClassWeights(num_classes=3)
    node.statistical_initialization(iter(stream))
    expected = WeightedCrossEntropyLoss.inverse_frequency_weights(labels, 3)
    assert node._weights.tolist() == pytest.approx(expected)


def test_class_weights_estimator_drops_ignore_index():
    # ignore_index targets must not inflate any class count.
    node = InverseFrequencyClassWeights(num_classes=2, ignore_index=-100)
    node.statistical_initialization(iter([{"targets": torch.tensor([0, 0, 1, -100, -100])}]))
    expected = WeightedCrossEntropyLoss.inverse_frequency_weights(torch.tensor([0, 0, 1]), 2)
    assert node._weights.tolist() == pytest.approx(expected)


def test_class_weights_estimator_port_contract():
    node = InverseFrequencyClassWeights(num_classes=3)
    node.statistical_initialization(iter([{"targets": torch.tensor([0, 1, 2])}]))
    out = node.forward(targets=torch.tensor([0, 1, 2]))["class_weights"]
    assert out.shape == (3,)
    assert out.dtype == torch.float32


def test_class_weights_estimator_forward_before_fit_raises():
    node = InverseFrequencyClassWeights(num_classes=3)
    assert node.requires_initial_fit is True  # the trainer will run Phase 1 for this node
    with pytest.raises(RuntimeError, match="statistical_initialization"):
        node.forward(targets=torch.tensor([0, 1, 2]))


def test_class_weights_estimator_empty_stream_raises():
    node = InverseFrequencyClassWeights(num_classes=3, ignore_index=-100)
    with pytest.raises(RuntimeError, match="no valid targets"):
        node.statistical_initialization(iter([{"targets": torch.full((4,), -100)}]))


def test_class_weights_estimator_serializes():
    # Fitted weights + the fitted flag travel through state_dict (buffer, not a plain attribute).
    labels = torch.tensor([0, 0, 0, 1, 2, 2])
    node = InverseFrequencyClassWeights(num_classes=3)
    node.statistical_initialization(iter([{"targets": labels}]))
    fresh = InverseFrequencyClassWeights(num_classes=3)
    assert bool(fresh._fitted) is False
    fresh.load_state_dict(node.state_dict())
    assert bool(fresh._fitted) is True
    assert fresh._weights.tolist() == pytest.approx(node._weights.tolist())


def test_loss_class_weights_port_overrides_buffer():
    # The class_weights input port takes precedence over the constructor buffer.
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    targets = torch.tensor([0, 3])
    node = WeightedCrossEntropyLoss(num_classes=4, class_weights=[1.0, 1.0, 1.0, 1.0])
    buffered = node.forward(logits=logits, targets=targets)["loss"].item()
    ported = node.forward(
        logits=logits, targets=targets, class_weights=torch.tensor([1.0, 2.0, 3.0, 4.0])
    )["loss"].item()
    assert ported != pytest.approx(buffered)  # the port weights changed the result
    ref = WeightedCrossEntropyLoss(num_classes=4, class_weights=[1.0, 2.0, 3.0, 4.0])
    assert ported == pytest.approx(ref.forward(logits=logits, targets=targets)["loss"].item())


def test_pipeline_feeds_fitted_weights_to_loss():
    # End to end: the estimator's fitted weights reach the loss through the class_weights port.
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

    k = 3
    labels = torch.tensor([0, 0, 0, 1, 2, 2])
    estimator = InverseFrequencyClassWeights(name="cw", num_classes=k)
    estimator.statistical_initialization(iter([{"targets": labels}]))  # Phase-1 fit
    loss = WeightedCrossEntropyLoss(name="loss", num_classes=k)
    pipe = CuvisPipeline("weights_to_loss")
    pipe.connect((estimator.outputs.class_weights, loss.inputs.class_weights))

    torch.manual_seed(0)
    logits = torch.randn(6, k)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    out = pipe.forward(batch={"logits": logits, "targets": targets}, stage=ExecutionStage.TRAIN)

    counts = torch.bincount(labels, minlength=k)
    weights = inverse_frequency_from_counts(counts, k).to(torch.float32)
    ref = WeightedCrossEntropyLoss(num_classes=k).forward(
        logits=logits, targets=targets, class_weights=weights
    )["loss"]
    assert torch.allclose(out[(loss.name, "loss")], ref)


# --- MulticlassSegmentationMetrics -------------------------------------------

_CTX = Context(stage=ExecutionStage.TEST, epoch=0, batch_idx=0)


def _as_dict(metrics):
    return {m.name: m.value for m in metrics}


def test_metrics_perfect_via_logits():
    node = MulticlassSegmentationMetrics(num_classes=3)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    out = node.forward(targets=targets, context=_CTX, logits=_confident_logits(targets, 3))
    d = _as_dict(out["metrics"])
    assert d["pixel_accuracy"] == pytest.approx(1.0)
    assert d["macro_precision"] == pytest.approx(1.0)
    assert d["macro_recall"] == pytest.approx(1.0)
    assert d["macro_f1"] == pytest.approx(1.0)


def test_metrics_accepts_predictions():
    node = MulticlassSegmentationMetrics(num_classes=3)
    targets = torch.tensor([0, 1, 2])
    out = node.forward(targets=targets, context=_CTX, predictions=torch.tensor([0, 1, 2]))
    assert _as_dict(out["metrics"])["pixel_accuracy"] == pytest.approx(1.0)


def test_metrics_requires_logits_or_predictions():
    node = MulticlassSegmentationMetrics(num_classes=3)
    with pytest.raises(ValueError, match="logits.*predictions"):
        node.forward(targets=torch.tensor([0, 1]), context=_CTX)


def test_metrics_ignore_index_masked():
    node = MulticlassSegmentationMetrics(num_classes=3, ignore_index=-100)
    targets = torch.tensor([0, 1, -100])
    preds = torch.tensor([0, 1, 2])  # the 3rd (ignored) would be wrong but is dropped
    out = node.forward(targets=targets, context=_CTX, predictions=preds)
    assert _as_dict(out["metrics"])["pixel_accuracy"] == pytest.approx(1.0)


def test_metrics_absent_class_excluded_from_macro():
    # num_classes=3 but class 2 never appears in targets -> excluded from macro.
    node = MulticlassSegmentationMetrics(num_classes=3)
    targets = torch.tensor([0, 0, 1, 1])
    out = node.forward(targets=targets, context=_CTX, predictions=torch.tensor([0, 0, 1, 1]))
    d = _as_dict(out["metrics"])
    assert d["macro_recall"] == pytest.approx(1.0)  # only classes 0,1 averaged
    assert "recall_class_2" not in d  # absent class not reported
    assert d["recall_class_0"] == pytest.approx(1.0)


def test_metrics_accumulate_and_reset_on_epoch():
    node = MulticlassSegmentationMetrics(num_classes=2)
    c0 = Context(stage=ExecutionStage.TEST, epoch=0, batch_idx=0)
    c1 = Context(stage=ExecutionStage.TEST, epoch=0, batch_idx=1)
    # batch 0: all correct; batch 1: all wrong -> running accuracy 0.5 over the epoch
    node.forward(targets=torch.tensor([0, 1]), context=c0, predictions=torch.tensor([0, 1]))
    out = node.forward(targets=torch.tensor([0, 1]), context=c1, predictions=torch.tensor([1, 0]))
    assert _as_dict(out["metrics"])["pixel_accuracy"] == pytest.approx(0.5)
    # new epoch resets the confusion -> only the fresh (correct) batch counts
    c_new = Context(stage=ExecutionStage.TEST, epoch=1, batch_idx=0)
    out2 = node.forward(
        targets=torch.tensor([0, 1]), context=c_new, predictions=torch.tensor([0, 1])
    )
    assert _as_dict(out2["metrics"])["pixel_accuracy"] == pytest.approx(1.0)


def test_metrics_macro_f1_is_mean_of_per_class_f1():
    # class 0: precision 1.0, recall 2/3 -> F1 0.8 ; class 1: precision 0.5, recall 1.0 -> F1 2/3.
    # macro_f1 averages the per-class F1s (0.8, 2/3); it is NOT the F1 of macro precision/recall.
    node = MulticlassSegmentationMetrics(num_classes=2)
    targets = torch.tensor([0, 0, 0, 1])
    preds = torch.tensor([0, 0, 1, 1])
    expected = (0.8 + 2 / 3) / 2
    d = _as_dict(node.forward(targets=targets, context=_CTX, predictions=preds)["metrics"])
    assert d["macro_f1"] == pytest.approx(expected, abs=1e-4)
    assert node.epoch_metrics()["macro_f1"] == pytest.approx(expected, abs=1e-4)
