import { CONFIG } from '../config';
import type { GatingConfig } from './fsm';

/** Welcher Detektor die Detektionen erzeugt hat — bestimmt die Gate-Kalibrierung.
 *  OVD-Scores (LLMDet/grounding) liegen systematisch tiefer als der COCO-Closed-Set-RT-DETR,
 *  darum sind die Schwellen keine universelle Konstante, sondern ein Per-Detektor-Parameter. */
export type DetectorKind = 'ovd' | 'rtdetr';

/** Aktive Gates in-place auf den Detektor umstellen. Die FSM hält eine Referenz auf `target` und
 *  liest cfg.tauFull/tauSoft erst zur Dispatch-Zeit — darum mutieren statt neu bauen, sonst gingen
 *  FSM-Zustand und Hypothese beim Detektorwechsel verloren. Die CONFIG-Presets bleiben unberührt. */
export function applyGatesForDetector(target: GatingConfig, detector: DetectorKind): void {
  const src = detector === 'ovd' ? CONFIG.gatingOvd : CONFIG.gating;
  target.tauFull = src.tauFull;
  target.tauSoft = src.tauSoft;
  target.armedTimeoutMs = src.armedTimeoutMs;
}
