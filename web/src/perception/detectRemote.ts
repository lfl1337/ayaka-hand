// web/src/perception/detectRemote.ts
import type { DetectorKind } from '../control/gating';
import type { Detection } from './ranking';

export interface RemoteInferResult {
  detections: Detection[];
  detector?: DetectorKind;              // Server-Backend ('ovd' | 'rtdetr') — bestimmt die Gate-Kalibrierung; fehlt bei Alt-Servern
}

/** Encode the canvas as JPEG and POST it raw to the studio server's /infer.
 *  Returns absolute-pixel detections (same `Detection` contract as the local WASM detect())
 *  plus the server's detector backend, so the caller can calibrate the gates for it.
 *  Throws on any transport/HTTP error → analyzeSnapshot's catch turns it into fail-safe no_grasp. */
export async function detectRemote(canvas: HTMLCanvasElement, endpoint: string): Promise<RemoteInferResult> {
  const blob = await canvasToJpegBlob(canvas);
  const res = await fetch(`${endpoint}/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'image/jpeg' },
    body: blob,
    signal: AbortSignal.timeout(10_000), // hängender /infer darf die Snapshot-Inputs nicht dauerhaft sperren (busy-Flag)
  });
  if (!res.ok) throw new Error(`infer HTTP ${res.status}`);
  return extractResult(await res.json());
}

/** Accept either a bare detections array or an envelope `{ detections: [...], detector?: ... }`. */
export function extractResult(json: unknown): RemoteInferResult {
  if (Array.isArray(json)) return { detections: json as Detection[] };
  if (json && typeof json === 'object' && Array.isArray((json as { detections?: unknown }).detections)) {
    const env = json as { detections: Detection[]; detector?: unknown };
    const detector = env.detector === 'ovd' || env.detector === 'rtdetr' ? env.detector : undefined;
    return { detections: env.detections, detector };
  }
  return { detections: [] };
}

function canvasToJpegBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('canvas.toBlob returned null'))),
      'image/jpeg',
      0.9,
    );
  });
}
