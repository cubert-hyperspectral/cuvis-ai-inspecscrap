# Metal-scrap CNN3D results (framework path)

Trained via `restore-trainrun`-style GradientTrainer (PyTorch Lightning), 68 epoch(s), Adam lr=1e-3, weighted CE, random-frame split (seed 42), 7x7 patches.

| metric | this run | paper Table 2 |
|---|---:|---:|
| pixel accuracy | 73.62% | 76.47% |
| macro precision | 73.53% | 83.56% |
| macro recall | 79.50% | 69.59% |

## Per-class recall (test)

| class | recall |
|---|---:|
| steel | 80.19% |
| aluminium | 82.56% |
| dark_rust_metal | 95.63% |
| painted_metal | 82.82% |
| stone | 96.78% |
| wood | 96.56% |
| painted_wood | 98.24% |
| plastic | 63.30% |
| rubber | 34.64% |
| styropor | 64.29% |

## Notes
- Framework training path: nodes wired into a CuvisPipeline, optimised by GradientTrainer; patches served by MetalScrapPatchDataModule (cached). Provisional 29->14 merge; random-frame split. A 1-epoch run is a smoke test, not the reproduction number.

