# D-FINE-Small detector weights (self-hosted)

The quantized ONNX weights (~11 MB) are **bundled in this repository** so a fresh clone runs
offline. To re-fetch them from Hugging Face instead:

```bash
cd web && bash scripts/fetch-detector.sh
```

Source: `onnx-community/dfine_s_coco-ONNX` (Apache-2.0, `DFineForObjectDetection`). Lands under
`dfine_s_coco/` (config.json, preprocessor_config.json, onnx/model_quantized.onnx). Drop-in
replacement for RT-DETRv2 — same 640×640 input, same `RTDetrImageProcessor`, same closed-set
COCO-80 head (logits[1,300,80] + boxes[1,300,4]) — at ~48.5 AP vs. ~46. RT-DETRv2 stays on disk
under `../rtdetr/` as fallback. Verify headless with `node scripts/smoke-detector.mjs`.
