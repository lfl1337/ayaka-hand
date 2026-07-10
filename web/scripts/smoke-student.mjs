// web/scripts/smoke-student.mjs — einmaliger Beweis, dass student.onnx im ORT-WASM läuft.
// Aus web/: node scripts/smoke-student.mjs  (erwartet public/models/student/student.onnx)
import * as ort from 'onnxruntime-web';
import { readFileSync } from 'node:fs';

const buf = readFileSync('public/models/student/student.onnx');
const session = await ort.InferenceSession.create(buf.buffer, { executionProviders: ['wasm'] });
const x = new Float32Array(3 * 160 * 160).map(() => Math.random());
const out = await session.run({ image: new ort.Tensor('float32', x, [1, 3, 160, 160]) });
const g = out.grip_logits.data, f = out.force_logits.data;
if (g.length !== 5 || f.length !== 2) throw new Error(`dims ${g.length}/${f.length}`);
console.log('SMOKE OK — grip_logits:', Array.from(g, (v) => v.toFixed(3)).join(' '));
