# Metal-scrap CNN3D results (framework path, per-object 80/20 (paper protocol))

Trained via `restore-trainrun`-style GradientTrainer (PyTorch Lightning), 57 epoch(s), Adam lr=1e-3, weighted CE, per-object 80/20 (paper protocol) (seed 42), 7x7 patches.

| metric | this run | paper Table 2 |
|---|---:|---:|
| pixel accuracy | 62.91% | 76.47% |
| macro precision | 59.06% | 83.56% |
| macro recall | 65.14% | 69.59% |

## Per-class recall (test)

| class | recall |
|---|---:|
| steel | 62.73% |
| aluminium | 82.72% |
| dark_rust_metal | 85.34% |
| painted_metal | 81.87% |
| can | 59.47% |
| stone | 81.58% |
| wood | 99.24% |
| painted_wood | 63.24% |
| plastic | 36.79% |
| rubber | 23.83% |
| styropor | 43.75% |
| fabric | 61.09% |

## Notes
- Framework training path: nodes wired into a CuvisPipeline, optimised by GradientTrainer; patches served from a cached patch pool. Provisional 29->14 merge; per-object 80/20 (paper protocol).

