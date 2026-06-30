# Metal-scrap CNN2D results (framework path, per-object 80/20 (paper protocol))

Trained via `restore-trainrun`-style GradientTrainer (PyTorch Lightning), 36 epoch(s), Adam lr=1e-3, weighted CE, per-object 80/20 (paper protocol) (seed 42), 7x7 patches.

| metric | this run | paper Table 2 |
|---|---:|---:|
| pixel accuracy | 59.87% | 74.06% |
| macro precision | 61.08% | 83.00% |
| macro recall | 63.96% | 67.97% |

## Per-class recall (test)

| class | recall |
|---|---:|
| steel | 51.43% |
| aluminium | 80.15% |
| dark_rust_metal | 77.32% |
| painted_metal | 86.10% |
| can | 45.50% |
| stone | 95.77% |
| wood | 99.24% |
| painted_wood | 66.89% |
| plastic | 29.85% |
| rubber | 24.06% |
| styropor | 60.41% |
| fabric | 50.76% |

## Notes
- Framework training path: nodes wired into a CuvisPipeline, optimised by GradientTrainer; patches served by MetalScrapPatchDataModule (cached). Provisional 29->14 merge; random-frame split. A 1-epoch run is a smoke test, not the reproduction number.

