# Prediction-mask cleanup (3D-CNN, full test set)

Off-the-shelf cuvis-ai node-catalog cleanup nodes applied to the raw per-pixel 3D-CNN prediction over every test frame. Pixel metrics on labeled pixels.

| stage | pixel acc | macro P | macro R |
|---|---:|---:|---:|
| raw prediction | 79.39% | 73.06% | 80.15% |
| MaskRobustifier (per-class morphology) | 83.47% | 77.67% | 84.06% |

The MaskRobustifier cleanup node is part of the Cuvis.AI node catalog.

