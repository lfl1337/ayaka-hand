#!/usr/bin/env bash
# web/scripts/fetch-detector.sh — einmalig; lädt die onnx-community-Exporte der Browser-Detektoren
# nach public/models/. Aus web/ ausführen: `bash scripts/fetch-detector.sh`.
#   Primär (Judge-Demo):  D-FINE-Small  -> public/models/dfine/dfine_s_coco/
#   Fallback (auf Platte): RT-DETRv2-r18 -> public/models/rtdetr/rtdetr_v2_r18vd/
# Repo verifiziert 2026-07-08: onnx-community/dfine_s_coco-ONNX (Apache-2.0, DFineForObjectDetection,
# d_fine, 640x640, RTDetrImageProcessor) und onnx-community/rtdetr_v2_r18vd-ONNX (Apache-2.0).
set -euo pipefail

# Ein Export = config + preprocessor + quantisiertes ONNX, im Layout das transformers.js erwartet
# (<localModelPath>/<model_id>/…). Die q8-Pipeline lädt onnx/model_quantized.onnx.
fetch_model() {  # $1=HF-Repo  $2=Zielordner
  local repo="$1" dest="$2" base
  base="https://huggingface.co/$repo/resolve/main"
  mkdir -p "$dest/onnx"
  curl -fL "$base/config.json"               -o "$dest/config.json"
  curl -fL "$base/preprocessor_config.json"  -o "$dest/preprocessor_config.json"
  curl -fL "$base/onnx/model_quantized.onnx" -o "$dest/onnx/model_quantized.onnx"
  ls -lh "$dest" "$dest/onnx"
}

fetch_model onnx-community/dfine_s_coco-ONNX    public/models/dfine/dfine_s_coco
fetch_model onnx-community/rtdetr_v2_r18vd-ONNX public/models/rtdetr/rtdetr_v2_r18vd

# --- ORT-WASM-Runtime self-hosten (kein CDN beim Judge) ---
# @huggingface/transformers importiert 'onnxruntime-web/webgpu' und lädt zur Laufzeit den passenden
# WASM-Build: Safari/WebKit -> plain (ort-wasm-simd-threaded.{wasm,mjs}), sonst -> asyncify
# (ort-wasm-simd-threaded.asyncify.{wasm,mjs}). Wir kopieren beide Paare (Binary + .mjs-Loader) aus dem
# installierten onnxruntime-web nach public/ort/, damit ein frischer Clone OHNE CDN funktioniert.
# Quelle: onnxruntime-web/dist im node_modules (Version an @huggingface/transformers gepinnt — bei Bump neu ziehen).
ORT_DEST="public/ort"
ORT_DIST=""
for cand in node_modules/onnxruntime-web/dist node_modules/.pnpm/onnxruntime-web@*/node_modules/onnxruntime-web/dist; do
  [ -d "$cand" ] && ORT_DIST="$cand" && break
done
if [ -z "$ORT_DIST" ]; then
  echo "FEHLER: onnxruntime-web/dist nicht gefunden — zuerst 'pnpm install' im web/-Ordner ausführen." >&2
  exit 1
fi
mkdir -p "$ORT_DEST"
for f in ort-wasm-simd-threaded.wasm ort-wasm-simd-threaded.mjs \
         ort-wasm-simd-threaded.asyncify.wasm ort-wasm-simd-threaded.asyncify.mjs; do
  cp -f "$ORT_DIST/$f" "$ORT_DEST/$f"
done
echo "ORT-Runtime kopiert aus $ORT_DIST ->"
ls -lh "$ORT_DEST"
