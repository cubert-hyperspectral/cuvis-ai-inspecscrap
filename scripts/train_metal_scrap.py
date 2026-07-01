"""Train a metal-scrap patch classifier through the cuvis-ai framework trainers.

This is the framework training path (no hand-rolled loop): it wires the real nodes into a
``CuvisPipeline`` (classifier -> WeightedCrossEntropyLoss + MulticlassSegmentationMetrics), serves
the paper's 7x7 patch batches via ``MetalScrapPatchDataModule``, runs ``StatisticalTrainer`` (a no-op
here, no node needs an initial fit) then ``GradientTrainer`` (PyTorch Lightning) for the optimisation,
and reads the epoch-level test metrics off the metric node's accumulated confusion matrix. The same
graph + data module are what ``restore-trainrun`` drives from ``configs/trainrun/metal_scrap_*.yaml``.

Smoke (1 epoch, bounded test frames):
  PYTHONPATH=<plugin> <venv-py> scripts/train_metal_scrap.py --model mlp --epochs 1 --max-test-frames 6
Full run: drop --max-test-frames and raise --epochs (e.g. 25).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_core.training.config import (
    OptimizerConfig,
    PipelineMetadata,
    TrainerConfig,
)

from cuvis_ai_inspecscrap.data import MetalScrapPatchDataModule
from cuvis_ai_inspecscrap.node.classification.cnn2d import SpatialSpectralCNN2D
from cuvis_ai_inspecscrap.node.classification.cnn3d import SpectralSpatialCNN3D
from cuvis_ai_inspecscrap.node.classification.mlp import SpectralMLPClassifier
from cuvis_ai_inspecscrap.node.labels import PAPER_CLASSES
from cuvis_ai_inspecscrap.node.losses import WeightedCrossEntropyLoss
from cuvis_ai_inspecscrap.node.metrics import MulticlassSegmentationMetrics
from cuvis_ai_inspecscrap.node.normalization import PerChannelStandardizer

_DEFAULT_ROOT = "data/HSIMetalScrap"
_NUM_CLASSES = len(PAPER_CLASSES)
_TABLE2 = {  # paper Table 2 (pixel acc / macro precision / macro recall), %
    "mlp": (59.99, 75.53, 45.45),
    "cnn2d": (74.06, 83.00, 67.97),
    "cnn3d": (76.47, 83.56, 69.59),
}


class LossHistory(pl.Callback):
    """Record per-epoch train/val loss so convergence (and any plateau) is inspectable."""

    def __init__(self) -> None:
        self.history: list[dict[str, float | int | None]] = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Append ``{epoch, train_loss, val_loss}`` (skipping the sanity-check pass)."""
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics

        def g(k: str) -> float | None:
            v = m.get(k)
            return float(v) if v is not None else None

        self.history.append(
            {
                "epoch": int(trainer.current_epoch),
                "train_loss": g("train_loss"),
                "val_loss": g("val_loss"),
            }
        )


def _build_classifier(model: str):
    if model == "mlp":
        return SpectralMLPClassifier(name="classifier", in_channels=437, num_classes=_NUM_CLASSES)
    if model == "cnn2d":
        return SpatialSpectralCNN2D(name="classifier", in_channels=437, num_classes=_NUM_CLASSES)
    if model == "cnn3d":
        return SpectralSpatialCNN3D(name="classifier", in_channels=437, num_classes=_NUM_CLASSES)
    raise ValueError(f"unknown model {model!r}")


def _write_report(
    model: str, md: dict[str, float], epochs: int, out_dir: Path, split: str = "frame"
) -> Path:
    acc, prec, rec = (
        100 * md.get("pixel_accuracy", 0.0),
        100 * md.get("macro_precision", 0.0),
        100 * md.get("macro_recall", 0.0),
    )
    p_acc, p_prec, p_rec = _TABLE2[model]
    split_label = "per-object 80/20 (paper protocol)" if split == "object" else "random-frame 80/20"
    suffix = "_objsplit" if split == "object" else ""
    lines = [
        f"# Metal-scrap {model.upper()} results (framework path, {split_label})\n",
        f"Trained via `restore-trainrun`-style GradientTrainer (PyTorch Lightning), {epochs} epoch(s), "
        f"Adam lr=1e-3, weighted CE, {split_label} (seed 42), 7x7 patches.\n",
        "| metric | this run | paper Table 2 |",
        "|---|---:|---:|",
        f"| pixel accuracy | {acc:.2f}% | {p_acc:.2f}% |",
        f"| macro precision | {prec:.2f}% | {p_prec:.2f}% |",
        f"| macro recall | {rec:.2f}% | {p_rec:.2f}% |",
        "\n## Per-class recall (test)\n",
        "| class | recall |",
        "|---|---:|",
    ]
    for name in PAPER_CLASSES:
        if f"recall_{name}" in md:
            lines.append(f"| {name} | {100 * md[f'recall_{name}']:.2f}% |")
    lines.append(
        "\n## Notes\n"
        "- Framework training path: nodes wired into a CuvisPipeline, optimised by GradientTrainer; "
        f"patches served from a cached patch pool. Provisional 29->14 merge; {split_label}.\n"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"metal_scrap_{model}{suffix}_framework_results.md"
    path.write_text("\n".join(lines) + "\n", "utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp", choices=["mlp", "cnn2d", "cnn3d"])
    ap.add_argument(
        "--split",
        default="frame",
        choices=["frame", "object"],
        help="frame = whole-frame 80/20; object = per-piece 80/20 (paper protocol).",
    )
    ap.add_argument("--root", default=_DEFAULT_ROOT)
    ap.add_argument("--patch-size", type=int, default=7)
    ap.add_argument("--train-samples", type=int, default=600)
    ap.add_argument("--val-samples", type=int, default=400)
    ap.add_argument("--eval-cap", type=int, default=5000)
    ap.add_argument("--max-test-frames", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="EarlyStopping patience on val_loss; 0 disables (fixed epochs).",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()

    dm_cls = MetalScrapPatchDataModule
    if args.split == "object":
        from cuvis_ai_inspecscrap.data.metal_scrap_object_patch_datamodule import (
            MetalScrapObjectPatchDataModule,
        )

        dm_cls = MetalScrapObjectPatchDataModule
    dm = dm_cls(
        root=args.root,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        eval_cap=args.eval_cap,
        max_test_frames=args.max_test_frames,
        seed=args.seed,
    )
    dm.setup(stage="fit")
    train_labels = dm.train_ds.labels
    print(f"[data] train={len(dm.train_ds)} val={len(dm.val_ds)} patches ({time.time() - t0:.0f}s)")

    in_channels = int(dm.train_ds.patches.shape[-1])
    standardizer = PerChannelStandardizer(name="standardizer", in_channels=in_channels)
    classifier = _build_classifier(args.model)
    loss = WeightedCrossEntropyLoss(
        name="loss",
        num_classes=_NUM_CLASSES,
        class_weights=WeightedCrossEntropyLoss.inverse_frequency_weights(
            train_labels, _NUM_CLASSES
        ),
    )
    metrics = MulticlassSegmentationMetrics(
        name="metrics", num_classes=_NUM_CLASSES, class_names=list(PAPER_CLASSES)
    )

    pipeline = CuvisPipeline(f"Metal_Scrap_{args.model.upper()}")
    pipeline.connect(
        (standardizer.normalized, classifier.patches),
        (classifier.logits, loss.logits),
        (classifier.logits, metrics.logits),
    )

    # Phase 1: fit the per-band standardizer's global mean/std (StatisticalTrainer).
    nodes = (standardizer, classifier, loss, metrics)
    if any(node.requires_initial_fit for node in nodes):
        print("[phase 1] StatisticalTrainer: fitting per-band standardization stats ...")
        StatisticalTrainer(pipeline=pipeline, datamodule=dm).fit()

    # Phase 2: gradient training.
    pipeline.unfreeze_nodes_by_name(["classifier"])
    trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    print(f"[model] {args.model} trainable params: {trainable:,}")

    out_dir = Path(__file__).resolve().parents[1] / "reports"
    models_dir = out_dir / "models"
    metrics_dir = out_dir / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_objsplit" if args.split == "object" else ""
    ckpt_dir = out_dir / "ckpt" / f"{args.model}{suffix}"
    loss_cb = LossHistory()
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir), monitor="val_loss", mode="min", save_top_k=1, filename="best"
    )
    callbacks: list[pl.Callback] = [loss_cb, ckpt_cb]
    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", mode="min", patience=args.early_stop_patience, min_delta=1e-4
            )
        )

    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=dm,
        loss_nodes=[loss],
        metric_nodes=[metrics],
        callbacks=callbacks,
        trainer_config=TrainerConfig(
            max_epochs=args.epochs,
            accelerator=args.device if args.device != "auto" else "auto",
            devices=1,
            enable_checkpointing=True,
            enable_progress_bar=True,
            log_every_n_steps=10,
        ),
        optimizer_config=OptimizerConfig(name="adam", lr=args.lr),
    )
    grad_trainer.fit()

    # Persist the per-epoch loss curve, then load the best-val-loss weights before testing/saving.
    (metrics_dir / f"metal_scrap_{args.model}{suffix}_losshist.json").write_text(
        json.dumps(loss_cb.history), "utf-8"
    )
    best_path = ckpt_cb.best_model_path
    stopped_early = bool(loss_cb.history) and loss_cb.history[-1]["epoch"] + 1 < args.epochs
    if best_path and os.path.exists(best_path):
        grad_trainer.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
        print(f"[ckpt] loaded best val_loss model: {best_path}")
    grad_trainer.test(ckpt_path=None)

    # Authoritative epoch-level metrics from the confusion matrix (not Lightning's batch-mean).
    md = metrics.epoch_metrics()
    best_val = min(
        (h["val_loss"] for h in loss_cb.history if h["val_loss"] is not None), default=None
    )
    print(
        f"[converge] epochs_run={len(loss_cb.history)} stopped_early={stopped_early} "
        f"best_val_loss={best_val}"
    )
    report = _write_report(args.model, md, len(loss_cb.history), metrics_dir, args.split)

    pipeline.save_to_file(
        str(models_dir / f"metal_scrap_{args.model}{suffix}_framework.yaml"),
        metadata=PipelineMetadata(
            name=f"Metal_Scrap_{args.model.upper()}",
            description=f"{args.model} patch classifier, framework-trained ({args.epochs} epochs).",
            tags=["metal_scrap", args.model],
        ),
    )
    print(
        f"[done] pixel_acc={100 * md['pixel_accuracy']:.2f}% "
        f"macro_P={100 * md['macro_precision']:.2f}% macro_R={100 * md['macro_recall']:.2f}% "
        f"-> {report} (total {time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
