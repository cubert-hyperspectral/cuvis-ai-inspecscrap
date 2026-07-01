"""Multi-class segmentation metrics: pixel accuracy + macro precision/recall/F1 + per-class recall.

Pure-torch confusion matrix (``bincount(K * target + pred)``) accumulated across batches within a
``(stage, epoch)`` and reset at the boundary, so the value emitted on the last batch is the
epoch-level metric (the AnomalyDetectionMetrics running-state idiom). Macro precision/recall average
only over classes present in the targets (``zero_division`` exclusion), matching the paper's
reporting where a split may not contain every class.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context, Metric
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class MulticlassSegmentationMetrics(Node):
    """Pixel accuracy + macro precision/recall/F1 + per-class recall from logits or predictions."""

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.CLASSIFICATION})

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, "num_classes"),
            description="Class logits [N, num_classes]; argmaxed internally (optional)",
            optional=True,
        ),
        "predictions": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            description="Integer class predictions [N] (optional; provide this or logits)",
            optional=True,
        ),
        "targets": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            description="Integer class targets [N]; ignore_index entries are dropped",
        ),
    }
    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -100,
        class_names: list[str] | None = None,
        execution_stages: set[ExecutionStage] | None = None,
        **kwargs: Any,
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.class_names = list(class_names) if class_names is not None else None
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(
            name=name,
            execution_stages=execution_stages,
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_names=self.class_names,
            **kwargs,
        )
        self.register_buffer(
            "_confusion", torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)
        )
        self._last_key: tuple[Any, int] | None = None

    def _class_label(self, i: int) -> str:
        if self.class_names is not None and i < len(self.class_names):
            return self.class_names[i]
        return f"class_{i}"

    @torch.no_grad()
    def epoch_metrics(self) -> dict[str, float]:
        """Authoritative epoch-level metrics from the accumulated confusion matrix.

        Call this after an eval epoch to read the true cumulative metric. Lightning's default
        logging means the per-batch values this node emits (each a running cumulative), which
        biases a cumulative metric; this reads the full confusion directly instead. ``GradientTrainer``
        logs the per-batch values to TensorBoard for monitoring; for the reported number, use this.
        """
        cm = self._confusion.to(torch.float64)
        diag = torch.diagonal(cm)
        row_sum, col_sum, total = cm.sum(dim=1), cm.sum(dim=0), cm.sum()
        present = row_sum > 0
        recall_c = torch.where(row_sum > 0, diag / row_sum.clamp_min(1), torch.zeros_like(diag))
        precision_c = torch.where(col_sum > 0, diag / col_sum.clamp_min(1), torch.zeros_like(diag))
        pr_sum = precision_c + recall_c
        f1_c = torch.where(
            pr_sum > 0, 2 * precision_c * recall_c / pr_sum.clamp_min(1e-12), torch.zeros_like(diag)
        )
        out = {
            "pixel_accuracy": float(diag.sum() / total) if total > 0 else 0.0,
            "macro_precision": float(precision_c[present].mean()) if bool(present.any()) else 0.0,
            "macro_recall": float(recall_c[present].mean()) if bool(present.any()) else 0.0,
            "macro_f1": float(f1_c[present].mean()) if bool(present.any()) else 0.0,
        }
        for i in range(self.num_classes):
            if bool(present[i]):
                out[f"recall_{self._class_label(i)}"] = float(recall_c[i])
        return out

    @torch.no_grad()
    def forward(
        self,
        targets: Tensor,
        context: Context,
        logits: Tensor | None = None,
        predictions: Tensor | None = None,
    ) -> dict[str, Any]:
        """Accumulate the confusion matrix for this batch and emit running metrics."""
        if logits is None and predictions is None:
            raise ValueError(
                "MulticlassSegmentationMetrics needs either 'logits' or 'predictions'."
            )
        if predictions is not None:
            preds = predictions.reshape(-1).to(torch.long)
        else:
            preds = logits.reshape(-1, self.num_classes).argmax(dim=-1).to(torch.long)
        tgt = targets.reshape(-1).to(torch.long)

        valid = tgt != self.ignore_index
        preds, tgt = preds[valid], tgt[valid]

        key = (context.stage, context.epoch)
        if self._last_key != key:
            self._confusion.zero_()
            self._last_key = key

        k = self.num_classes
        if tgt.numel():
            idx = (tgt * k + preds).to(self._confusion.device)
            self._confusion += torch.bincount(idx, minlength=k * k).reshape(k, k)

        cm = self._confusion.to(torch.float64)
        diag = torch.diagonal(cm)
        row_sum = cm.sum(dim=1)  # support per true class
        col_sum = cm.sum(dim=0)  # predicted per class
        total = cm.sum()

        pixel_acc = (diag.sum() / total).item() if total > 0 else 0.0
        present = row_sum > 0  # classes that appear in targets
        recall_c = torch.where(row_sum > 0, diag / row_sum.clamp_min(1), torch.zeros_like(diag))
        precision_c = torch.where(col_sum > 0, diag / col_sum.clamp_min(1), torch.zeros_like(diag))
        n_present = int(present.sum())
        macro_recall = (recall_c[present].mean().item()) if n_present else 0.0
        macro_precision = (precision_c[present].mean().item()) if n_present else 0.0
        pr_sum = precision_c + recall_c
        f1_c = torch.where(
            pr_sum > 0, 2 * precision_c * recall_c / pr_sum.clamp_min(1e-12), torch.zeros_like(diag)
        )
        macro_f1 = (f1_c[present].mean().item()) if n_present else 0.0

        def m(name: str, value: float) -> Metric:
            return Metric(
                name=name,
                value=float(value),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )

        metrics = [
            m("pixel_accuracy", pixel_acc),
            m("macro_precision", macro_precision),
            m("macro_recall", macro_recall),
            m("macro_f1", macro_f1),
        ]
        # Per-class recall for present classes (drives the copper-confusion read).
        for i in range(k):
            if bool(present[i]):
                metrics.append(m(f"recall_{self._class_label(i)}", recall_c[i].item()))
        return {"metrics": metrics}


__all__ = ["MulticlassSegmentationMetrics"]
