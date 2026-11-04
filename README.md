# cuvis-ai-inspecscrap

[![CI](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/actions/workflows/ci.yml/badge.svg)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai-inspecscrap/branch/main/graph/badge.svg)](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai-inspecscrap)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A [Cuvis.AI](https://github.com/cubert-hyperspectral/cuvis-ai) reproduction of the InSpecScrap
SWIR hyperspectral steel-scrap pixel classifier from Gursch et al. (2026). The paper's MLP, 2D-CNN,
and 3D-CNN classifiers, its class-weighted cross-entropy loss, its pixel/macro metrics, and the
object-level majority vote are rebuilt as Cuvis.AI framework nodes and trained with the framework's
own `StatisticalTrainer` and `GradientTrainer`, with no bespoke training loop.

## Results

Built on the public InSpecScrap release (170 SWIR cubes, 437 bands, 10 of the paper's 14 classes
present in the data). The full write-up, with figures, is in
[`reports/report_cuvis_ai_inspecscrap.html`](reports/report_cuvis_ai_inspecscrap.html).

| Model | Pixel accuracy (whole-frame split) | Paper Table 2 |
|---|---:|---:|
| MLP | 62.3% | 59.99% |
| 2D-CNN | 62.4% | 74.06% |
| 3D-CNN (best) | 71.5% | 76.47% |

The 3D-CNN reproduces the paper's pixel accuracy on a whole-frame split and reaches 62.9% under a
stricter leak-safe per-piece split. The object-level majority vote lifts dense accuracy from 77.3%
to 83.6%, and an off-the-shelf `MaskRobustifier` cleanup stage from the Cuvis.AI node catalog lifts
the full-test prediction from 77.3% to 80.4% with no retraining. The paper's model ranking
(3D-CNN > 2D-CNN > MLP) holds throughout.

## What is in the plugin

A single plugin manifest, [`configs/plugins/cuvis_ai_inspecscrap.yaml`](configs/plugins/cuvis_ai_inspecscrap.yaml),
exposes the node and data-module family:

- **Data modules** (`metal_scrap`, `metal_scrap_patch`, `metal_scrap_dense_patch`): read the SWIR
  TIFF cubes and paired colorized label PNGs and serve the paper's 7x7 spatial-spectral patches.
- **`RgbLabelToClassIndex`**: fail-closed mapping from the release's colorized label PNGs to the
  paper's integer class taxonomy.
- **`PatchesToCube`**: renames the patch data modules' `patches` batch key to `cube` so it feeds
  cuvis-ai's built-in `Standardize` node, which does the paper's global per-band z-score (fitted in
  the statistical phase); the pipelines wire `patches -> PatchesToCube -> Standardize -> classifier`.
- **`PatchSampler`**, **`SpectralMLPClassifier`**, **`SpatialSpectralCNN2D`**,
  **`SpectralSpatialCNN3D`**: the patch sampler and the three classifiers.
- **`WeightedCrossEntropyLoss`**: inverse-frequency class-weighted multi-class cross-entropy.
- **`MulticlassSegmentationMetrics`**: pixel accuracy and macro precision/recall from a confusion
  matrix.
- **`BlobMajorityVote`**, **`ClassMapAccumulator`**, **`MontageColumnSink`**: object-level pooling
  and dense-inference / report assembly sinks.

The pipelines under `configs/pipeline/` and the trained configs under `reports/models/` also list
`cuvis_ai_builtin` in their `plugins:` and reference `cuvis_ai.node.pretreatments.standardize.Standardize`
by name. This repo's own runtime code (the node and data-module classes) never imports `cuvis-ai`;
the host that runs these pipelines (`restore-pipeline`, the gRPC orchestrator) supplies it, and it is
a dev/scripts dependency here (see `pyproject.toml`), not a runtime one.

## Dataset

The InSpecScrap HSI metal-scrap dataset is published on Zenodo and is **not redistributed here**.
Download it and place it under `data/HSIMetalScrap` (the default `root` in the trainrun configs):

> Jaschik, Jernej. InSpecScrap hyperspectral steel-scrap dataset (HSI Metal Scrap).
> Zenodo. DOI [10.5281/zenodo.17076238](https://doi.org/10.5281/zenodo.17076238)

## Install

This plugin's own runtime code depends only on `cuvis-ai-core` and `cuvis-ai-schemas`, never the
high-level [`cuvis-ai`](https://github.com/cubert-hyperspectral/cuvis-ai) package. Running a
pipeline needs `cuvis-ai` installed in the host env (it supplies `cuvis_ai_builtin`'s `Standardize`
node that the pipelines reference), which `restore-pipeline` and `restore-trainrun` already assume:

```bash
uv pip install -e .
uv pip install "cuvis-ai>=0.21.0"   # or: uv sync --extra dev / --extra training
```

## Reproduce

After downloading the dataset to `data/HSIMetalScrap`, train any of the three models through the
framework trainers:

```bash
# declarative: drive the trainrun config through restore-trainrun. The pipeline lists
# cuvis_ai_builtin (for Standardize), so pass its plugins dir alongside this repo's own.
restore-trainrun \
  --trainrun-path configs/trainrun/metal_scrap_cnn3d.yaml \
  --plugins-dir   configs/plugins \
  --plugins-dir   "$(python -c 'import cuvis_ai, os; print(os.path.join(os.path.dirname(cuvis_ai.__file__), "configs", "plugins"))')" \
  --mode          train

# or scripted, with a leak-safe per-piece split (needs the `training` extra: uv sync --extra training)
python scripts/train_metal_scrap.py --model cnn3d --split object --epochs 200
```

Swap `cnn3d` for `cnn2d` or `mlp` to train the other models. `--split frame` uses the whole-frame
split; `--split object` uses the leak-safe per-piece split. Trained pipelines and metrics are
written under `reports/models/` and `reports/metrics/`.

## Acknowledgements and citation

This repository reproduces published work. The method and the dataset are the work of the
InSpecScrap consortium (Know-Center, TU Graz, JOANNEUM RESEARCH, K1-MET); please cite the original
paper and dataset:

> Gursch, Ofner, Harb, Jaschik, Ganster, Rieger (2026).
> Hyperspectral scrap characterisation for scrap composition optimisation in steel recycling.
> *Waste Management & Research.* DOI [10.1177/0734242X261451604](https://doi.org/10.1177/0734242X261451604)
> (open access).

```bibtex
@article{gursch2026hyperspectral,
  title   = {Hyperspectral scrap characterisation for scrap composition optimisation in steel recycling},
  author  = {Gursch and Ofner and Harb and Jaschik and Ganster and Rieger},
  journal = {Waste Management \& Research},
  year    = {2026},
  doi     = {10.1177/0734242X261451604}
}
```

Deviations from the paper (a provisional 29-to-14 label-colour merge, a connected-component proxy
for the object-level vote, an inverse-frequency loss weighting, and 10 of 14 classes present in the
public release) are described in the report.

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE). The InSpecScrap
dataset is distributed separately on Zenodo under its own license; this repository neither includes
nor relicenses it.
