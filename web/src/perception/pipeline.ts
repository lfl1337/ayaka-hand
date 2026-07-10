// web/src/perception/pipeline.ts
import type { DetectorKind } from '../control/gating';
import type { GraspHypothesis } from '../types';
import { getRemoteEndpoint } from '../remote';
import { detect } from './detector';
import { detectRemote } from './detectRemote';
import { lookupGrip } from './lookup';
import { type Detection, rankDetections } from './ranking';
import { inferGrip, studentReady } from './student';

/** Gerankte Detektionen → Hypothese. Der pure Kern (rank läuft EINMAL upstream, entschieden wird hier).
 *  Geteilt von hypothesisFromDetections und analyzeDetailed → Verhalten lebt an genau einer Stelle. */
function hypothesisFromRanked(ranked: Detection[]): GraspHypothesis {
  if (!ranked.length) return { grip: 'no_grasp', force: 'firm', confidence: 0.95, source: 'edge', objectLabel: 'nothing detected' };
  const top = ranked[0];
  const m = lookupGrip(top.label);
  return {
    grip: m.grip, force: m.force,
    confidence: Math.min(1, top.score * m.conf),
    source: 'edge', objectLabel: top.label,
    contactPoint: { x: 0.5, y: 0.5 },
  };
}

/** Detektionen → Hypothese: rank → lookup → GraspHypothesis. Die EINE Quelle der Wahrheit —
 *  lokaler WASM-Snapshot, Studio-/infer und der SSE-Frame-Pfad teilen sich genau diese Logik.
 *  Pur (kein I/O), damit sie testbar bleibt und das Verhalten nur an einer Stelle lebt. */
export function hypothesisFromDetections(dets: Detection[], w: number, h: number): GraspHypothesis {
  return hypothesisFromRanked(rankDetections(dets, w, h));
}

/** Wie hypothesisFromDetections, liefert zusätzlich die gerankte Liste + den gewählten Index —
 *  damit die UI zeichnen kann, WAS erkannt wurde (Tracking-Overlay). rankDetections läuft genau einmal. */
export interface DetailedHypothesis {
  hypothesis: GraspHypothesis;
  ranked: Detection[];
  chosenIdx: number;                    // Index des gewählten Objekts in `ranked` (0 = Top-Rank; -1 wenn leer)
  detector?: DetectorKind;              // welches Backend die Detektionen erzeugt hat → Gate-Kalibrierung (nur analyzeSnapshot setzt es)
}
export function analyzeDetailed(dets: Detection[], w: number, h: number): DetailedHypothesis {
  const ranked = rankDetections(dets, w, h);
  return { hypothesis: hypothesisFromRanked(ranked), ranked, chosenIdx: ranked.length ? 0 : -1 };
}

/** Wie analyzeDetailed, aber der Top-Kandidat bekommt Grip/Kraft vom destillierten CNN
 *  (Pixel statt Label-Lookup), wenn der Student geladen ist. Fällt bei JEDEM Fehler auf
 *  den Lookup-Pfad zurück — die FSM sieht nie einen Unterschied im Fehlerfall. */
export async function analyzeDetailedWithStudent(
  dets: Detection[], w: number, h: number, canvas: HTMLCanvasElement,
): Promise<DetailedHypothesis> {
  const base = analyzeDetailed(dets, w, h);
  if (base.chosenIdx < 0 || !studentReady()) {
    return { ...base, hypothesis: { ...base.hypothesis, via: 'lookup' } };
  }
  const top = base.ranked[base.chosenIdx];
  const cnn = await inferGrip(canvas, top.box);
  if (!cnn) return { ...base, hypothesis: { ...base.hypothesis, via: 'lookup' } };
  return {
    ...base,
    hypothesis: {
      ...base.hypothesis,
      grip: cnn.grip, force: cnn.force,
      confidence: Math.min(1, top.score * cnn.gripProb),
      via: 'cnn',
    },
  };
}

/** Snapshot → Detailergebnis. Wirft NIE: jeder Fehlerpfad liefert no_grasp (fail-safe).
 *  Ist ?remote= gesetzt, geht das Bild an den Studio-Server /infer statt ins lokale WASM;
 *  beide Wege münden in denselben analyzeDetailed-Pfad. */
export async function analyzeSnapshot(canvas: HTMLCanvasElement): Promise<DetailedHypothesis> {
  try {
    const endpoint = getRemoteEndpoint();
    if (endpoint) {
      const remote = await detectRemote(canvas, endpoint);
      return { ...await analyzeDetailedWithStudent(remote.detections, canvas.width, canvas.height, canvas), detector: remote.detector ?? 'rtdetr' }; // Alt-Server ohne Feld waren rtdetr-only (gleiche Backward-Compat wie SSE-Frames)
    }
    const dets = await detect(canvas);
    return { ...await analyzeDetailedWithStudent(dets, canvas.width, canvas.height, canvas), detector: 'rtdetr' }; // lokales WASM ist immer Closed-Set
  } catch (err) {
    console.error('analyzeSnapshot failed → fail-safe no_grasp', err);
    return {
      hypothesis: { grip: 'no_grasp', force: 'firm', confidence: 1, source: 'edge', objectLabel: 'error', via: 'lookup' },
      ranked: [], chosenIdx: -1,
    };
  }
}
