# Metal-scrap 3D-CNN full-test evaluation (pixel + object-level)

Full test split (all frames), every labeled pixel densely classified. 2457296 test pixels, 10 classes present: aluminium, dark_rust_metal, painted_metal, painted_wood, plastic, rubber, steel, stone, styropor, wood.

Object-level = BlobMajorityVote (connected components, conn=2) over each frame's foreground, the paper's inference-time vote.

### Pixel-level

| metric | value |
|---|---:|
| pixel accuracy | 79.39% |
| macro precision | 73.06% |
| macro recall | 80.15% |

| class | recall |
|---|---:|
| steel | 83.55% |
| aluminium | 84.75% |
| dark_rust_metal | 96.22% |
| painted_metal | 89.77% |
| stone | 96.39% |
| wood | 96.06% |
| painted_wood | 98.46% |
| plastic | 54.89% |
| rubber | 37.57% |
| styropor | 63.86% |

### Object-level (majority vote)

| metric | value |
|---|---:|
| pixel accuracy | 86.44% |
| macro precision | 86.29% |
| macro recall | 82.87% |

| class | recall |
|---|---:|
| steel | 97.19% |
| aluminium | 100.00% |
| dark_rust_metal | 100.00% |
| painted_metal | 99.83% |
| stone | 100.00% |
| wood | 30.07% |
| painted_wood | 100.00% |
| plastic | 43.47% |
| rubber | 58.19% |
| styropor | 100.00% |

### Copper-confusion read

Copper test pixels: 0. Where copper is predicted (paper: 23.4% correct, ~38% -> steel):
  - copper absent from test set

### Paper Table 2 (3D-CNN, 14-class, 200 ep): pixel 76.47% / macro-P 83.56% / macro-R 69.59%.
