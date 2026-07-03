# Metal-scrap 3D-CNN full-test evaluation (pixel + object-level)

Full test split (all frames), every labeled pixel densely classified. 2457296 test pixels, 10 classes present: aluminium, dark_rust_metal, painted_metal, painted_wood, plastic, rubber, steel, stone, styropor, wood.

Object-level = BlobMajorityVote (connected components, conn=2) over each frame's foreground, the paper's inference-time vote.

### Pixel-level

| metric | value |
|---|---:|
| pixel accuracy | 77.29% |
| macro precision | 71.48% |
| macro recall | 78.02% |

| class | recall |
|---|---:|
| steel | 77.36% |
| aluminium | 82.11% |
| dark_rust_metal | 96.67% |
| painted_metal | 89.03% |
| stone | 94.13% |
| wood | 97.95% |
| painted_wood | 96.72% |
| plastic | 44.07% |
| rubber | 30.02% |
| styropor | 72.14% |

### Object-level (majority vote)

| metric | value |
|---|---:|
| pixel accuracy | 83.63% |
| macro precision | 83.14% |
| macro recall | 77.42% |

| class | recall |
|---|---:|
| steel | 82.35% |
| aluminium | 100.00% |
| dark_rust_metal | 100.00% |
| painted_metal | 100.00% |
| stone | 100.00% |
| wood | 30.07% |
| painted_wood | 100.00% |
| plastic | 43.44% |
| rubber | 18.30% |
| styropor | 100.00% |

### Copper-confusion read

Copper test pixels: 0. Where copper is predicted (paper: 23.4% correct, ~38% -> steel):
  - copper absent from test set

### Paper Table 2 (3D-CNN, 14-class, 200 ep): pixel 76.47% / macro-P 83.56% / macro-R 69.59%.
