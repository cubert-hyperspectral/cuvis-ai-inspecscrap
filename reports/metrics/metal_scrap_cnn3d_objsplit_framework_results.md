# Metal-scrap CNN3D results (framework path, per-object 80/20 (paper protocol))

Trained via `restore-trainrun`-style GradientTrainer (PyTorch Lightning), 45 epoch(s), Adam lr=1e-3, weighted CE, per-object 80/20 (paper protocol) (seed 42), 7x7 patches.

| metric | this run | paper Table 2 |
|---|---:|---:|
| pixel accuracy | 62.85% | 76.47% |
| macro precision | 60.01% | 83.56% |
| macro recall | 66.20% | 69.59% |

## Per-class recall (test)

| class | recall |
|---|---:|
| steel | 60.20% |
| aluminium | 79.72% |
| dark_rust_metal | 79.36% |
| painted_metal | 85.30% |
| can | 62.13% |
| stone | 82.39% |
| wood | 98.68% |
| painted_wood | 70.14% |
| plastic | 31.30% |
| rubber | 35.54% |
| styropor | 62.39% |
| fabric | 47.31% |

## Notes
- Framework training path: nodes wired into a CuvisPipeline, optimised by GradientTrainer; patches served by MetalScrapPatchDataModule (cached). Provisional 29->14 merge; random-frame split. A 1-epoch run is a smoke test, not the reproduction number.

