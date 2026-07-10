<!-- Numbers are the project's source of truth. Do not recompute, round, or overwrite
     from an auto-generated table — the ablation section below is hand-curated. -->
# Distillation: before / after (validation n=613)

| Metric | Lookup baseline (2017-style strawman, perfect detector) | Student (2.23M CNN, pixels only) |
|---|---|---|
| Grip accuracy | 70.5 % | **72.3 %** |
| Grip macro-F1 | 0.581 | **0.659** |
| Force accuracy | 65.4 % | **81.1 %** |
| Force macro-F1 | 0.595 | **0.806** |

| Grip class | Baseline F1 | Student F1 |
|---|---|---|
| power | 0.722 | 0.752 |
| lateral | 0.766 | 0.702 |
| pinch | 0.000 | 0.323 |
| tripod | 0.594 | 0.727 |
| no_grasp | 0.823 | 0.790 |

_Reference = teacher labels (Qwen3-VL-32B, Prompt v1.1). The baseline is handed the ground-truth
class name (the strongest possible strawman); the student sees only pixels._

## Ablation (same val set, same pipeline)

| Run | Architecture | Params | Init | Epochs | Grip Acc | Grip macro-F1 | Force Acc | Force macro-F1 |
|---|---|---|---|---|---|---|---|---|
| kd-v1 | MobileNetV2-0.75 | 1.36M | scratch | 80 | 49.9 % | 0.484 | 74.4 % | 0.741 |
| kd-v2 | MobileNetV2-0.75 | 1.36M | scratch | 200 | — | 0.497* | — | — |
| **kd-v3 (chosen)** | **MobileNetV2-1.0 (timm)** | **2.23M** | **ImageNet** | **80** | **72.3 %** | **0.659** | **81.1 %** | **0.806** |
| kd-v4 | MobileNetV2-0.5 (timm) | 0.70M | ImageNet | 80 | 69.5 % | 0.635 | 81.6 % | 0.812 |

\* kd-v2: best val grip macro-F1 from the training log; no full eval, since it was below baseline.

**Finding:** from-scratch loses to the lookup strawman on grip (11k images are too few for
from-scratch); ImageNet init flips it. The student beats the baseline exactly where the baseline is
structurally blind: `pinch` (lookup F1 0.0) and **force/state** (+16 points of accuracy) — the core
thesis "sees form AND state" is measured. Even the 0.70M model (kd-v4) beats the baseline on grip
macro-F1.

Training: RTX 5090, bf16, weighted CE (grip + 0.5·force), AdamW, cosine, 11,004 teacher labels
(COCO crops, license-filtered), Prompt v1.1. ONNX export kd-v3: 8.9 MB, ORT parity Δ ≈ 6e-6.

**Parameter count:** the communicated figure is **2,232,839 ≈ 2.23M** (timm `mobilenetv2_100.ra_in1k`,
ImageNet-initialized). History note: an earlier "2.6M" wrongly counted the discarded ImageNet
classifier head; the from-scratch 0.75-width model was 1.36M and lost to the baseline. The old
figures are not reused.
