# Metal-scrap MLP results (framework path, random-frame 80/20)

Trained via `restore-trainrun`-style GradientTrainer (PyTorch Lightning), 69 epoch(s), Adam lr=1e-3, weighted CE, random-frame 80/20 (seed 42), 7x7 patches.

| metric | this run | paper Table 2 |
|---|---:|---:|
| pixel accuracy | 62.31% | 59.99% |
| macro precision | 66.64% | 75.53% |
| macro recall | 68.23% | 45.45% |

## Per-class recall (test)

| class | recall |
|---|---:|
| steel | 58.74% |
| aluminium | 65.59% |
| dark_rust_metal | 74.33% |
| painted_metal | 75.72% |
| stone | 85.87% |
| wood | 93.74% |
| painted_wood | 97.07% |
| plastic | 44.76% |
| rubber | 10.33% |
| styropor | 76.11% |

## Notes
- Framework training path: nodes wired into a CuvisPipeline, optimised by GradientTrainer; patches served from a cached patch pool. Provisional 29->14 merge; random-frame 80/20.

