# Metal-scrap MLP results (framework path, per-object 80/20 (paper protocol))

Trained via `restore-trainrun`-style GradientTrainer (PyTorch Lightning), 104 epoch(s), Adam lr=1e-3, weighted CE, per-object 80/20 (paper protocol) (seed 42), 7x7 patches.

| metric | this run | paper Table 2 |
|---|---:|---:|
| pixel accuracy | 51.87% | 59.99% |
| macro precision | 49.95% | 75.53% |
| macro recall | 55.13% | 45.45% |

## Per-class recall (test)

| class | recall |
|---|---:|
| steel | 40.02% |
| aluminium | 68.48% |
| dark_rust_metal | 62.87% |
| painted_metal | 79.84% |
| can | 48.06% |
| stone | 83.95% |
| wood | 98.02% |
| painted_wood | 66.08% |
| plastic | 34.56% |
| rubber | 18.17% |
| styropor | 60.82% |
| fabric | 0.73% |

## Notes
- Framework training path: nodes wired into a CuvisPipeline, optimised by GradientTrainer; patches served from a cached patch pool. Provisional 29->14 merge; per-object 80/20 (paper protocol).

