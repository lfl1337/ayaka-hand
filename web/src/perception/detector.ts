// web/src/perception/detector.ts
import { env, pipeline, type ObjectDetectionPipeline, type ProgressInfo } from '@huggingface/transformers';
import { CONFIG } from '../config';
import { ORT_NUM_THREADS, ortWasmPaths } from './ortEnv';
import type { Detection } from './ranking';

export type DetectorState = 'idle' | 'loading' | 'ready' | 'failed';

/** Live-Ladezustand des Detektors — von der UI gepollt (Busy-Badge) und per Listener gepusht.
 *  `msg` trägt bei state==='failed' die reale Fehlermeldung (sonst nie surfacen → Live-Gate blind). */
export const detectorStatus: { state: DetectorState; pct?: number; msg?: string } = { state: 'idle' };

type ProgressListener = (status: { state: DetectorState; pct?: number; msg?: string }) => void;
let onDetectorProgress: ProgressListener | null = null;

/** UI registriert sich hier, um Lade-/Fehlerzustände live in die Badge zu spiegeln. */
export function setDetectorProgressListener(cb: ProgressListener | null): void {
  onDetectorProgress = cb;
}

function setStatus(state: DetectorState, pct?: number, msg?: string): void {
  detectorStatus.state = state;
  detectorStatus.pct = pct;
  detectorStatus.msg = msg;
  onDetectorProgress?.({ state, pct, msg });
}

let detPromise: Promise<ObjectDetectionPipeline> | null = null;

/** D-FINE-Small (Apache-2.0) via transformers.js, WASM-Backend, Modell-Dateien SELF-HOSTED
 *  unter /models/dfine/ (kein CDN-Zugriff beim Judge). Drop-in-Swap von RT-DETRv2: gleicher 640×640-Input,
 *  gleicher DFineForObjectDetection-Kopf (logits[1,300,80]+boxes[1,300,4]) → nur Pfad + Pipeline-Id geändert.
 *  rtdetr bleibt als Fallback unter /models/rtdetr/ auf Platte. Bei Ladefehler wird die gecachte Promise
 *  auf null zurückgesetzt → ein Retry startet frisch (kein hängender Reject-Cache, Task-7-Review). */
export function loadDetector(): Promise<ObjectDetectionPipeline> {
  if (detPromise) return detPromise;
  env.allowRemoteModels = false;
  env.allowLocalModels = true;
  env.localModelPath = CONFIG.detector.modelPath;
  // ORT-WASM-Runtime SELF-HOSTED unter /ort/ (kein CDN beim Judge): transformers.js zeigt wasmPaths
  // sonst auf jsdelivr → offline ein sofortiger Init-Fail bei 0 % (genau das Live-Gate-Symptom).
  // Datei-Wahl + Thread-Baseline leben GETEILT in ortEnv.ts (identisch mit student.ts, kein Drift).
  const wasm = env.backends.onnx.wasm as {
    wasmPaths?: string | { wasm: string; mjs: string };
    numThreads?: number;
  };
  wasm.wasmPaths = ortWasmPaths();
  wasm.numThreads = ORT_NUM_THREADS;
  setStatus('loading', 0);
  const p = (pipeline('object-detection', 'dfine_s_coco', {
    dtype: 'q8',                // → onnx/model_quantized.onnx (~11 MB int8), wie beim rtdetr-Export
    device: 'wasm',              // explizit pinnen: transformers v4 würde auf Handys sonst WebGPU auto-wählen (und dort failen)
    // transformers.js feuert je Datei {status, progress?} — nur die Prozent-Events tragen progress.
    progress_callback: (info: ProgressInfo) => {
      if ('progress' in info) setStatus('loading', Math.round(info.progress));
    },
  }) as Promise<ObjectDetectionPipeline>).then(
    (det) => { setStatus('ready', 100); return det; },
    (err: unknown) => {
      // Reale Fehlermeldung sichern (nie mehr schlucken) → Badge + Telemetrie; danach Cache leeren für frischen Retry.
      setStatus('failed', undefined, String((err as { message?: unknown })?.message ?? err).slice(0, 300));
      detPromise = null;
      throw err;
    },
  );
  detPromise = p;
  return p;
}

export async function detect(canvas: HTMLCanvasElement): Promise<Detection[]> {
  const det = await loadDetector();
  const url = canvas.toDataURL('image/jpeg', 0.9);
  const raw = (await det(url, { threshold: CONFIG.detector.scoreThreshold })) as
    { label: string; score: number; box: { xmin: number; ymin: number; xmax: number; ymax: number } }[];
  return raw.map((r) => ({ label: r.label, score: r.score, box: r.box }));
}
