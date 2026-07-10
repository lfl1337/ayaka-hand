// web/src/perception/student.ts
// Der destillierte 2,2M-Student im Browser: ONNX via onnxruntime-web (WASM self-hosted
// unter /ort/ — dieselben Runtime-Dateien wie der Detektor, kein CDN beim Judge).
// Ausfall-Semantik: JEDER Fehler => null; der Aufrufer fällt auf die Lookup zurück.
import * as ort from 'onnxruntime-web';
import type { Force, Grip } from '../types';
import type { Detection } from './ranking';
import { IMAGENET_MEAN, IMAGENET_STD, STUDENT_FORCES, STUDENT_GRIPS, cropPlan, softmax } from './studentGeom';

const MODEL_URL = '/models/student/student.onnx';
const SIZE = 160;

// Self-hosted ORT-WASM (Judge-Regel: Demo darf nicht am CDN hängen). Der Student teilt sich den
// onnxruntime-web-Backend mit dem Detektor — welches Modul zuerst lädt, setzt die Pfade. Beide MÜSSEN
// exakt DIESELBE Datei-Wahl treffen (Safari plain vs. asyncify), sonst lädt der zweite Init einen
// inkompatiblen Build. Spiegelt detector.ts (isAppleWebkit + exakte /ort/-Dateipaare) bewusst wider.
function isAppleWebkit(): boolean {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  const isAppleVendor = (navigator.vendor || '').indexOf('Apple') > -1;
  const notOtherBrowser = !/CriOS|FxiOS|EdgiOS|OPiOS|mercury|brave/i.test(ua)
    && !ua.includes('Chrome') && !ua.includes('Android');
  return isAppleVendor && notOtherBrowser;
}
const ORT_BASE = '/ort/';
ort.env.wasm.wasmPaths = isAppleWebkit()
  ? { mjs: `${ORT_BASE}ort-wasm-simd-threaded.mjs`, wasm: `${ORT_BASE}ort-wasm-simd-threaded.wasm` }               // iPhone/WebKit: plain
  : { mjs: `${ORT_BASE}ort-wasm-simd-threaded.asyncify.mjs`, wasm: `${ORT_BASE}ort-wasm-simd-threaded.asyncify.wasm` }; // Android/Chrome: asyncify
ort.env.wasm.numThreads = 1;        // kein COOP/COEP nötig (statisches Hosting)

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let ready = false;

export function loadStudent(): Promise<void> {
  sessionPromise ??= ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] })
    .then((s) => { ready = true; return s; })                                // Dims-Check erfolgt beim ersten Run (Metadata-API variiert je ORT-Build)
    .catch((err) => { sessionPromise = null; ready = false; throw err; });   // Retry möglich (wie detector.ts)
  return sessionPromise.then(() => undefined);
}

export function studentReady(): boolean { return ready; }

export async function inferGrip(
  canvas: HTMLCanvasElement,
  box: Detection['box'],
): Promise<{ grip: Grip; force: Force; gripProb: number } | null> {
  try {
    // Crop SYNCHRON vor jedem await: pinnt die Pixel an das Frame, das beim Aufruf auf #snap lag
    // (JS ist single-threaded → zwischen drawImage und getImageData kann kein neueres Frame das
    // geteilte Canvas überschreiben). Erst danach auf die Session warten und inferieren.
    const off = document.createElement('canvas');
    off.width = SIZE; off.height = SIZE;
    const ctx = off.getContext('2d', { willReadFrequently: true })!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';                              // hochwertiges Downscaling → F1-Treue auf kleinen Crops
    const p = cropPlan(box, canvas.width, canvas.height);
    ctx.drawImage(canvas, p.sx, p.sy, p.sw, p.sh, p.dx, p.dy, p.dw, p.dh);
    const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
    const chw = new Float32Array(3 * SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      for (let c = 0; c < 3; c++) {
        chw[c * SIZE * SIZE + i] = (data[i * 4 + c] / 255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
      }
    }
    const session = await (sessionPromise ?? Promise.reject(new Error('student not loaded')));
    const out = await session.run({ image: new ort.Tensor('float32', chw, [1, 3, SIZE, SIZE]) });
    const gripLogits = out['grip_logits'].data as Float32Array;
    const forceLogits = out['force_logits'].data as Float32Array;
    if (gripLogits.length !== STUDENT_GRIPS.length || forceLogits.length !== STUDENT_FORCES.length) {
      throw new Error(`unexpected head dims ${gripLogits.length}/${forceLogits.length}`);
    }
    const gp = softmax(gripLogits); const fp = softmax(forceLogits);
    const gi = gp.indexOf(Math.max(...gp)); const fi = fp.indexOf(Math.max(...fp));
    return { grip: STUDENT_GRIPS[gi], force: STUDENT_FORCES[fi], gripProb: gp[gi] };
  } catch (err) {
    console.error('student inferGrip failed → lookup fallback', err);
    return null;
  }
}
