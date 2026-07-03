# Metal-scrap CNN3D results (framework path, random-frame 80/20)

Trained via `restore-trainrun`-style GradientTrainer (PyTorch Lightning), 52 epoch(s), Adam lr=1e-3, weighted CE, random-frame 80/20 (seed 42), 7x7 patches.

| metric | this run | paper Table 2 |
|---|---:|---:|
| pixel accuracy | 71.55% | 76.47% |
| macro precision | 72.85% | 83.56% |
| macro recall | 77.80% | 69.59% |

## Per-class recall (test)

| class | recall |
|---|---:|
| steel | 74.35% |
| aluminium | 77.86% |
| dark_rust_metal | 96.67% |
| painted_metal | 84.31% |
| stone | 94.25% |
| wood | 97.98% |
| painted_wood | 96.49% |
| plastic | 54.62% |
| rubber | 27.48% |
| styropor | 74.01% |

## Notes
- Framework training path: nodes wired into a CuvisPipeline, optimised by GradientTrainer; patches served from a cached patch pool. Provisional 29->14 merge; random-frame 80/20.

