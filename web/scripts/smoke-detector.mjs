// web/scripts/smoke-detector.mjs — headless proxy for the browser detect() verification.
// The perf gate lives in Task 10 / the browser; here we only prove the self-hosted detector
// export loads from public/models and produces plausible detections.
//
// EP note: transformers.js' Node build has NO 'wasm' device (only coreml/webgpu/cpu — all via
// onnxruntime-node). WASM is a browser-only backend. So this headless smoke runs the 'cpu' EP
// on the onnxruntime-node prebuilt binary (present in the tarball; the workspace no-build flag
// only blocks build *scripts*, not the shipped .node). This exercises the exact same model +
// pre/post-processing as the browser; only the execution provider differs. The deployed path
// (src/perception/detector.ts) sets no device and gets WASM in-browser — unaffected by this.
//
// Run from web/:  node scripts/smoke-detector.mjs [imagePath]
//   default image: /tmp/coco-cats.jpg (COCO val2017/000000039769.jpg — two cats, two remotes)
//   default model: D-FINE-Small (the deployed browser detector). Override via env for the fallback:
//     SMOKE_MODEL_PATH=./public/models/rtdetr/ SMOKE_MODEL_ID=rtdetr_v2_r18vd node scripts/smoke-detector.mjs

import { env, pipeline } from '@huggingface/transformers';

const IMG = process.argv[2] ?? '/tmp/coco-cats.jpg';
const MODEL_PATH = process.env.SMOKE_MODEL_PATH ?? './public/models/dfine/';
const MODEL_ID = process.env.SMOKE_MODEL_ID ?? 'dfine_s_coco';

env.allowRemoteModels = false;          // no CDN — same posture as the judge/edge
env.localModelPath = MODEL_PATH;

console.log(`[smoke] loading ${MODEL_ID} (q8, cpu EP) from ${env.localModelPath} ...`);
const t0 = performance.now();
const detector = await pipeline('object-detection', MODEL_ID, { dtype: 'q8', device: 'cpu' });
const tLoad = performance.now() - t0;
console.log(`[smoke] pipeline ready in ${tLoad.toFixed(0)} ms`);

const t1 = performance.now();
const out = await detector(IMG, { threshold: 0.4 });
const tInfer = performance.now() - t1;

console.log(`[smoke] inference on ${IMG} in ${tInfer.toFixed(0)} ms — ${out.length} detection(s):`);
for (const d of out) {
  const b = d.box;
  console.log(
    `  ${d.label.padEnd(14)} score=${d.score.toFixed(3)}  ` +
    `box=[${b.xmin.toFixed(0)}, ${b.ymin.toFixed(0)}, ${b.xmax.toFixed(0)}, ${b.ymax.toFixed(0)}]`,
  );
}

if (out.length === 0) {
  console.error('[smoke] FAIL: zero detections — model or preprocessing broken');
  process.exit(1);
}
console.log('[smoke] OK: plausible detections produced.');
