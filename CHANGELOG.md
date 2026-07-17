# Changelog

## 0.2.2 - 2026-07-17

- Require `cuvis-ai-schemas>=0.8.0` and `cuvis-ai-core>=0.11.2`, adopting the released cuvis-ai-next framework versions.
- **Corrected the README result numbers** to match the refreshed v0.2.1 report: 3D-CNN whole-frame
  pixel accuracy 71.5%, object-level majority vote 77.3% -> 83.6%, and `MaskRobustifier` cleanup
  77.3% -> 80.4%. The prior values were carried over from the pre-retrain report.

## 0.2.1 - 2026-07-03

- **Refreshed the reproduction report** from a fresh, from-scratch training run (all three models,
  both the whole-frame and leak-safe per-piece splits, 200-epoch cap with early stopping). The
  figures, metric tables, and model weights are regenerated, and the HTML report's tables and
  headline figures are now derived directly from the run's metrics rather than static values.
- **Registered `InverseFrequencyClassWeights`** in the plugin manifest so the Phase-1 class-weight
  estimator is discoverable alongside `WeightedCrossEntropyLoss`.

## 0.2.0 - 2026-06-30

Initial public release. A Cuvis.AI reproduction of the InSpecScrap SWIR steel-scrap pixel
classifier from Gursch et al. 2026.

- **Classifier family as Cuvis.AI nodes.** `SpectralMLPClassifier`, `SpatialSpectralCNN2D`, and
  `SpectralSpatialCNN3D` reproduce the paper's three architectures over 7x7 spatial-spectral
  patches, with `PatchSampler`, the fail-closed `RgbLabelToClassIndex` label mapper, the global
  per-band `PerChannelStandardizer`, the inverse-frequency `WeightedCrossEntropyLoss`, and
  `MulticlassSegmentationMetrics` (pixel accuracy, macro precision/recall).
- **Two-phase framework training.** Training runs through the framework's `StatisticalTrainer` and
  `GradientTrainer` from `configs/trainrun/metal_scrap_{mlp,cnn2d,cnn3d}.yaml`, with both a
  whole-frame and a leak-safe per-piece split. No bespoke training loop.
- **Object-level pooling.** `BlobMajorityVote` collapses the per-pixel prediction to one label per
  connected component; `ClassMapAccumulator` and `MontageColumnSink` scatter and assemble dense
  predictions for the report.
- **Data modules.** `metal_scrap`, `metal_scrap_patch`, and `metal_scrap_dense_patch` read the
  InSpecScrap SWIR TIFF cubes and paired colorized label PNGs and serve the paper's patch batches.
- **Single plugin manifest** at `configs/plugins/cuvis_ai_inspecscrap.yaml`; generic colourise and
  overlay viz nodes live in the Cuvis.AI node catalog.
- **Reproduction report** at `reports/report_cuvis_ai_inspecscrap.html`, with the full result
  tables, figures, and the deviations from the paper.
