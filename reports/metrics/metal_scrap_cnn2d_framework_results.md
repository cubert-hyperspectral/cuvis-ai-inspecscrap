# Metal-scrap CNN2D results (framework path)

Trained via `restore-trainrun`-style GradientTrainer (PyTorch Lightning), 31 epoch(s), Adam lr=1e-3, weighted CE, random-frame split (seed 42), 7x7 patches.

| metric | this run | paper Table 2 |
|---|---:|---:|
| pixel accuracy | 62.35% | 74.06% |
| macro precision | 67.06% | 83.00% |
| macro recall | 69.94% | 67.97% |

## Per-class recall (test)

| class | recall |
|---|---:|
| steel | 77.02% |
| aluminium | 73.98% |
| dark_rust_metal | 91.17% |
| painted_metal | 77.00% |
| stone | 92.13% |
| wood | 95.84% |
| painted_wood | 97.23% |
| plastic | 50.41% |
| rubber | 2.05% |
| styropor | 42.58% |

## Notes
- Framework training path: nodes wired into a CuvisPipeline, optimised by GradientTrainer; patches served by MetalScrapPatchDataModule (cached). Provisional 29->14 merge; random-frame split. A 1-epoch run is a smoke test, not the reproduction number.

