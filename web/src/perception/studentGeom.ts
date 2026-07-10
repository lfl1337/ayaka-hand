// web/src/perception/studentGeom.ts
// Pure Geometrie/Mathematik für den Studenten — spiegelt die VAL-Transform des Trainings:
// Resize(shorter->176) + CenterCrop(160) + ImageNet-Normalisierung. Abweichungen hier
// verfälschen die gemessene 0,659-F1-Zahl — nicht "vereinfachen".
import type { Force, Grip } from '../types';

export const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
export const IMAGENET_STD = [0.229, 0.224, 0.225] as const;
// Schema v1 (FROZEN) — identisch zur ONNX-Metadata des Exporters und factory/distill/distill/schema.py
export const STUDENT_GRIPS: Grip[] = ['power', 'lateral', 'pinch', 'tripod', 'no_grasp'];
export const STUDENT_FORCES: Force[] = ['delicate', 'firm'];

export interface CropPlacement {
  sx: number; sy: number; sw: number; sh: number;   // Quell-Rect (Box, auf Canvas geclampt)
  dw: number; dh: number;                            // Zielgröße (kürzere Seite = resizeShort)
  dx: number; dy: number;                            // Platzierung fürs zentrierte crop×crop-Fenster
}

export function cropPlan(
  box: { xmin: number; ymin: number; xmax: number; ymax: number },
  canvasW: number, canvasH: number,
  resizeShort = 176, crop = 160,
): CropPlacement {
  const sx = Math.max(0, Math.min(box.xmin, canvasW - 1));
  const sy = Math.max(0, Math.min(box.ymin, canvasH - 1));
  const sw = Math.max(1, Math.min(box.xmax, canvasW) - sx);
  const sh = Math.max(1, Math.min(box.ymax, canvasH) - sy);
  const scale = resizeShort / Math.min(sw, sh);
  const dw = sw * scale, dh = sh * scale;
  return { sx, sy, sw, sh, dw, dh, dx: (crop - dw) / 2, dy: (crop - dh) / 2 };
}

export function softmax(logits: Float32Array | number[]): number[] {
  const m = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}
