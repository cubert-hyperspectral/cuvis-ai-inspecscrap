# Metal-scrap label-sanity baseline

Per-class labeled-pixel distribution over the seeded random frame-level 80/20 split (split_mode=random_frame), with the provisional 29 -> 14 merge from `RgbLabelToClassIndex`. Ignored pixels (background, the `(255,204,51)` collision, and unmapped colours) are excluded from class counts.

Frames: train=116, val=20, test=34 (total 170).

| id | class | train px | val px | test px | total px | % of labeled |
|---|---|---:|---:|---:|---:|---:|
| 0 | steel | 1424521 | 95479 | 172709 | 1692709 | 13.365 |
| 1 | aluminium | 439869 | 35391 | 249833 | 725093 | 5.725 |
| 2 | copper | 590 | 0 | 0 | 590 | 0.005 |
| 3 | dark_rust_metal | 2563638 | 113030 | 38114 | 2714782 | 21.435 |
| 4 | light_rust_metal | 0 | 0 | 0 | 0 | 0.000 |
| 5 | painted_metal | 1418382 | 591919 | 900051 | 2910352 | 22.979 |
| 6 | can | 450422 | 0 | 0 | 450422 | 3.556 |
| 7 | stone | 216695 | 107528 | 62985 | 387208 | 3.057 |
| 8 | wood | 259411 | 6243 | 102131 | 367785 | 2.904 |
| 9 | painted_wood | 393568 | 0 | 170808 | 564376 | 4.456 |
| 10 | plastic | 883426 | 140546 | 368765 | 1392737 | 10.997 |
| 11 | rubber | 230462 | 187593 | 112439 | 530494 | 4.189 |
| 12 | styropor | 421216 | 182642 | 279461 | 883319 | 6.974 |
| 13 | fabric | 37345 | 8025 | 0 | 45370 | 0.358 |

Labeled pixels total: 12,665,237
Ignored pixels: train=28,379,815, val=4,931,604, test=8,422,704

## Majority-class floor (sanity)

Accuracy if always predicting the most-frequent labeled class, per split.

| split | majority class | floor accuracy |
|---|---|---:|
| train | dark_rust_metal | 29.33% |
| val | painted_metal | 40.31% |
| test | painted_metal | 36.63% |

## Notes

- A **dataset-level** split (DataSet0-2 train / DataSet3 test) was tried first and rejected: DataSet3 is ~68% `dark_rust_metal`, a class with ~0 pixels in DataSet0-2, while train-only classes are absent from DataSet3. The four DataSets are not comparable batches, so a random frame-level split is used to keep every class in train and test. This reinstates a file-level (not piece-level) leakage guard, pending JOANNEUM piece ids.
- Classes absent from the test split (excluded from macro metrics via zero_division): copper, light_rust_metal, can, fabric.
- The `light_rust_metal` class is ~0 px: its only source colour `(255,204,51)` is the `Me-LightRusty` / `Plastic_Packaging` collision, resolved to ignore pending the original JOANNEUM per-pixel class ids.
- Background dominates raw pixels (~75%) and is excluded here via ignore_index.
- This is the provisional merge; sign-off pending before final numbers.
