# Reports

Reproduction of Gursch et al. 2026 (SWIR hyperspectral steel-scrap classification) built on the
Cuvis.AI framework.

**Start here:** open [`report_cuvis_ai_inspecscrap.html`](report_cuvis_ai_inspecscrap.html), the
self-contained write-up (embedded figures, click any figure to enlarge).

## Layout

| Folder | Contents |
|---|---|
| `figures/` | Report figures (PNG): real-frame predictions, confusion matrices, per-class recall, split comparison, object-level recall, loss curves, mask cleanup. |
| `models/` | Trained pipelines per model and split: `*_framework.pt` (weights + confusion buffer) and `*_framework.yaml`, plus `objectlevel_confusion.pt` and `cleanup_confusions.pt`. |
| `metrics/` | Per-model metric reports (`*_framework_results.md`, frame and `_objsplit`), the object-level report, the mask-cleanup report, the label baseline, and per-run `*_losshist.json`. |

## Regenerate

Train and evaluate through the framework, which (re)populates `models/` and `metrics/`:

```bash
python scripts/train_metal_scrap.py --model {mlp,cnn2d,cnn3d} [--split {frame,object}] --epochs N
```

The mask-cleanup stage applies the `MaskRobustifier` node from the Cuvis.AI node catalog to the raw
3D-CNN prediction (no retraining).
