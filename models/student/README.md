# Student model (kd-v3)

Distilled grip/force classifier for the browser demo. `student.onnx` is **bundled in this
repository** so a fresh clone runs offline. It is fully reproducible from `factory/distill`
(train on the teacher labels, then export); see `web/scripts/fetch-student.sh`.

## Provenance & training

- **Backbone**: timm `mobilenetv2_100.ra_in1k` (Apache-2.0), ImageNet-initialized
- **Head**: dual-head — `grip` (5 classes) and `force` (2 classes)
- **Training data**: 11,004 license-filtered COCO crops (CC-BY family; attribution in the data
  manifests), labeled by Qwen3-VL-32B-Instruct (Prompt v1.1, Schema v1)
- **Parameters**: 2,232,839

## ONNX signature

- **Opset**: 17
- **Input**: `image` `[batch, 3, 160, 160]`, ImageNet-normalized
- **Outputs**: `grip_logits` `[., 5]` and `force_logits` `[., 2]`, in Schema-v1 order

## Validation (n=613)

| Metric | Student | Lookup baseline |
|---|---|---|
| Grip accuracy | 72.3 % | 70.5 % |
| Grip macro-F1 | 0.659 | 0.581 |
| Force accuracy | 81.1 % | 65.4 % |
| Force macro-F1 | 0.806 | 0.595 |

Details and ablation: `docs/benchmarks/distill-before-after.md`.
