# RT-DETRv2 detector weights (self-hosted)

The quantized ONNX weights (~20 MB) are **bundled in this repository** so a fresh clone runs
offline. To re-fetch them from Hugging Face instead:

```bash
cd web && bash scripts/fetch-detector.sh
```

Source: `onnx-community/rtdetr_v2_r18vd-ONNX` (Apache-2.0). Lands under
`rtdetr_v2_r18vd/` (config.json, preprocessor_config.json, onnx/model_quantized.onnx).
Verify headless with `node scripts/smoke-detector.mjs`.
