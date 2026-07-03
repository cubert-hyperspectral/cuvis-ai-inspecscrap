# Prediction-mask cleanup (3D-CNN, full test set)

Off-the-shelf cuvis-ai node-catalog cleanup nodes applied to the raw per-pixel 3D-CNN prediction over every test frame. Pixel metrics on labeled pixels.

| stage | pixel acc | macro P | macro R |
|---|---:|---:|---:|
| raw prediction | 77.29% | 71.48% | 78.02% |
| MaskRobustifier (per-class morphology) | 80.36% | 75.26% | 81.18% |

Node loaded from the cuvis-ai-extend-node-catalog checkout via the `cuvis_ai_extend_node_catalog` local-path plugin manifest.

