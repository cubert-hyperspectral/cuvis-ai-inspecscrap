"""Weighted multi-class cross-entropy loss + an inverse-frequency class-weight estimator.

``WeightedCrossEntropyLoss`` mirrors Gursch et al. 2026 Eq. 1 (class-frequency weighting). It
consumes ``logits`` [N, K] (leading dims flattened to N) + integer ``targets`` [N] and returns a
scalar loss; ``ignore_index`` drops background pixels. The class weights all come from the shared
``inverse_frequency_from_counts`` formula and can be supplied three ways: at construction
(``class_weights=``), applied to a live node (``set_class_weights`` / ``fit_class_weights``), or
through the optional ``class_weights`` input port. ``InverseFrequencyClassWeights`` drives that
port: an entry node that accumulates the training-split label distribution in
``statistical_initialization`` (fitted by ``StatisticalTrainer`` in Phase 1, like the per-band
standardizer), emitting the weights so the loss is weighted with no manual call.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


def inverse_frequency_from_counts(counts: Tensor, num_classes: int) -> Tensor:
    """Inverse-frequency class weights (Gursch et al. 2026 Eq. 1) from per-class counts.

    Returns a ``[num_classes]`` float tensor of ``total / (num_classes * count_c)`` per class, and
    ``0.0`` for any class with a zero count. Shared by the construction-time
    :meth:`WeightedCrossEntropyLoss.inverse_frequency_weights` and the Phase-1
    :class:`InverseFrequencyClassWeights` estimator, so both apply one formula.
    """
    counts = counts.to(torch.float64)
    total = counts.sum()
    return torch.where(
        counts > 0, total / (num_classes * counts.clamp_min(1)), torch.zeros_like(counts)
    )


class WeightedCrossEntropyLoss(Node):
    """Class-weighted multi-class cross-entropy over flattened logits + integer targets."""

    _category = NodeCategory.LOSS
    _tags = frozenset(
        {NodeTag.TRAINING, NodeTag.DIFFERENTIABLE, NodeTag.TORCH, NodeTag.CLASSIFICATION}
    )

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, "num_classes"),
            description="Class logits [N, num_classes] (leading dims flattened to N)",
        ),
        "targets": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            description="Integer class targets [N]; ignore_index entries are dropped",
        ),
        "class_weights": PortSpec(
            dtype=torch.float32,
            shape=("num_classes",),
            optional=True,
            description="Optional per-class weights [num_classes]; when connected they override "
            "the constructor weights (e.g. fed by InverseFrequencyClassWeights)",
        ),
    }
    OUTPUT_SPECS = {
        "loss": PortSpec(dtype=torch.float32, shape=(), description="Scalar weighted CE loss")
    }

    def __init__(
        self,
        num_classes: int,
        class_weights: list[float] | None = None,
        weight: float = 1.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if class_weights is not None and len(class_weights) != num_classes:
            raise ValueError(
                f"class_weights has {len(class_weights)} entries, expected num_classes={num_classes}"
            )
        self.num_classes = int(num_classes)
        self.weight = float(weight)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)
        # LossNode lives in the high-level `cuvis_ai` package (which eagerly imports the Cuvis
        # SDK), so this plugin stays on cuvis-ai-core and pins the loss stages itself: a loss
        # runs in TRAIN / VAL / TEST, never INFERENCE.
        kwargs.pop("execution_stages", None)
        super().__init__(
            num_classes=num_classes,
            class_weights=list(class_weights) if class_weights is not None else None,
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            execution_stages={ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST},
            **kwargs,
        )
        if class_weights is not None:
            self.register_buffer("_class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self._class_weights = None

    @staticmethod
    def inverse_frequency_weights(labels: Tensor, num_classes: int) -> list[float]:
        """Inverse-frequency class weights (Gursch et al. 2026 Eq. 1) from a label tensor.

        Returns ``total / (num_classes * count_c)`` per class and ``0.0`` for classes absent
        from ``labels``. Pass the result as ``class_weights`` at construction, or apply it to a
        live node via :meth:`set_class_weights` / :meth:`fit_class_weights`.
        """
        counts = torch.bincount(labels.reshape(-1).to(torch.long), minlength=num_classes)
        return inverse_frequency_from_counts(counts, num_classes).tolist()

    def set_class_weights(self, class_weights: list[float] | Tensor) -> None:
        """Set (or replace) the class-weight buffer that :meth:`forward` applies.

        The public counterpart to the ``class_weights`` constructor argument, for weights that
        are only known once the data is loaded. Registers/updates a buffer (not a plain attribute),
        so the weights travel with the node through ``.to(device)`` and serialization.
        """
        w = torch.as_tensor(class_weights, dtype=torch.float32).reshape(-1)
        if w.numel() != self.num_classes:
            raise ValueError(
                f"class_weights has {w.numel()} entries, expected num_classes={self.num_classes}"
            )
        if "_class_weights" in self._buffers:
            self._class_weights = w  # update the existing buffer in place
        else:
            if hasattr(self, "_class_weights"):
                del self._class_weights  # drop the None placeholder attribute
            self.register_buffer("_class_weights", w)

    def fit_class_weights(self, labels: Tensor) -> None:
        """Compute inverse-frequency weights from training labels and apply them to this node."""
        self.set_class_weights(self.inverse_frequency_weights(labels, self.num_classes))

    def forward(
        self, logits: Tensor, targets: Tensor, class_weights: Tensor | None = None, **_: Any
    ) -> dict[str, Tensor]:
        """Compute the (optionally class-weighted) cross-entropy over valid (non-ignored) targets.

        Weights come from the ``class_weights`` input port when connected, otherwise from the
        weights set at construction or via :meth:`set_class_weights`.
        """
        flat_logits = logits.reshape(-1, self.num_classes)
        flat_targets = targets.reshape(-1).to(torch.long)
        valid = flat_targets != self.ignore_index
        if not bool(valid.any()):
            # Keep the loss connected to the graph when a batch has no labeled pixels.
            return {"loss": 0.0 * flat_logits.sum()}
        cw = class_weights if class_weights is not None else self._class_weights
        if cw is not None:
            cw = cw.to(device=flat_logits.device, dtype=flat_logits.dtype)
        loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            weight=cw,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        return {"loss": self.weight * loss}


class InverseFrequencyClassWeights(Node):
    """Entry node that fits inverse-frequency class weights from the training-label distribution.

    Pairs with :class:`WeightedCrossEntropyLoss`: wire ``class_weights`` into the loss's
    ``class_weights`` port and the weighting becomes a Phase-1 statistic fitted by
    ``StatisticalTrainer`` (like the per-band standardizer's mean/std), with no manual call. As an
    entry node it receives ``targets`` straight from the batch; ``statistical_initialization``
    accumulates the per-class counts over the training stream and stores the frozen weights.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.TRAINING, NodeTag.TORCH, NodeTag.CLASSIFICATION})

    INPUT_SPECS = {
        "targets": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            optional=True,
            description="Integer class targets [N] from the batch (ignore_index dropped)",
        ),
    }
    OUTPUT_SPECS = {
        "class_weights": PortSpec(
            dtype=torch.float32,
            shape=("num_classes",),
            description="Inverse-frequency class weights [num_classes], fitted in Phase 1",
        ),
    }

    def __init__(self, num_classes: int, ignore_index: int = -100, **kwargs: Any) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        super().__init__(num_classes=num_classes, ignore_index=ignore_index, **kwargs)
        self.register_buffer("_weights", torch.ones(self.num_classes, dtype=torch.float32))
        self.register_buffer("_fitted", torch.zeros((), dtype=torch.bool))
        # Class weights are a data-derived statistic, so this node always needs the Phase-1 pass.
        self._requires_initial_fit_override = True

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Accumulate per-class label counts over the training stream, then store the weights."""
        counts = torch.zeros(self.num_classes, dtype=torch.float64)
        for batch_data in input_stream:
            t = batch_data.get("targets")
            if t is None:
                continue
            t = t.reshape(-1).to(torch.long)
            t = t[t != self.ignore_index]
            if t.numel() == 0:
                continue
            counts += torch.bincount(t, minlength=self.num_classes).to(torch.float64)
        if float(counts.sum()) == 0.0:
            raise RuntimeError(
                "InverseFrequencyClassWeights.statistical_initialization() saw no valid targets."
            )
        self._weights.copy_(inverse_frequency_from_counts(counts, self.num_classes))
        self._fitted.fill_(True)

    @torch.no_grad()
    def forward(self, targets: Tensor | None = None, **_: Any) -> dict[str, Tensor]:
        """Emit the fitted inverse-frequency weights (``targets`` is consumed only during fit)."""
        if not bool(self._fitted):
            raise RuntimeError(
                "InverseFrequencyClassWeights requires statistical_initialization() first."
            )
        return {"class_weights": self._weights}


__all__ = [
    "WeightedCrossEntropyLoss",
    "InverseFrequencyClassWeights",
    "inverse_frequency_from_counts",
]
