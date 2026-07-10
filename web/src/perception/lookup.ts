// web/src/perception/lookup.ts
import type { Force, Grip } from '../types';

/** COCO-Klasse → Griff. BEWUSST simpel: das ist die 2017er-Baseline („sieht Form, nicht Zustand")
 *  und ab Tag 4 der Ablation-Strohmann, den das destillierte CNN schlägt. */
const MAP: Record<string, { grip: Grip; force: Force; conf: number }> = {
  'cup':           { grip: 'power',   force: 'firm',     conf: 0.9 },
  'bottle':        { grip: 'power',   force: 'firm',     conf: 0.9 },
  'wine glass':    { grip: 'power',   force: 'delicate', conf: 0.85 },
  'bowl':          { grip: 'power',   force: 'firm',     conf: 0.8 },
  'banana':        { grip: 'power',   force: 'delicate', conf: 0.8 },
  'apple':         { grip: 'tripod',  force: 'firm',     conf: 0.85 },
  'orange':        { grip: 'tripod',  force: 'firm',     conf: 0.85 },
  'donut':         { grip: 'tripod',  force: 'delicate', conf: 0.8 },
  'cell phone':    { grip: 'lateral', force: 'firm',     conf: 0.9 },
  'book':          { grip: 'lateral', force: 'firm',     conf: 0.8 },
  'remote':        { grip: 'power',   force: 'firm',     conf: 0.8 },
  'scissors':      { grip: 'lateral', force: 'firm',     conf: 0.85 },
  'knife':         { grip: 'lateral', force: 'firm',     conf: 0.7 },
  'fork':          { grip: 'lateral', force: 'firm',     conf: 0.8 },
  'spoon':         { grip: 'lateral', force: 'firm',     conf: 0.8 },
  'toothbrush':    { grip: 'lateral', force: 'firm',     conf: 0.8 },
  'mouse':         { grip: 'power',   force: 'firm',     conf: 0.8 },
  'keyboard':      { grip: 'no_grasp', force: 'firm',    conf: 0.9 },
  'laptop':        { grip: 'no_grasp', force: 'firm',    conf: 0.9 },
  'sports ball':   { grip: 'power',   force: 'firm',     conf: 0.8 },
  'frisbee':       { grip: 'pinch',   force: 'firm',     conf: 0.75 },
  'teddy bear':    { grip: 'power',   force: 'delicate', conf: 0.8 },
  'vase':          { grip: 'power',   force: 'delicate', conf: 0.8 },
  'carrot':        { grip: 'tripod',  force: 'firm',     conf: 0.8 },
  // Open-Vocabulary-Zusatzklassen (LLMDet-Studio-Server) — exakt die classes.txt-Phrasen.
  'mug':           { grip: 'power',   force: 'firm',     conf: 0.9 },
  'drinking glass':{ grip: 'power',   force: 'delicate', conf: 0.85 },
  'pen':           { grip: 'lateral', force: 'firm',     conf: 0.85 },
  'key':           { grip: 'pinch',   force: 'firm',     conf: 0.85 },
  'headphone case':{ grip: 'power',   force: 'firm',     conf: 0.8 },
  'egg':           { grip: 'tripod',  force: 'delicate', conf: 0.9 },
  'plate':         { grip: 'lateral', force: 'delicate', conf: 0.75 },
  'jar':           { grip: 'power',   force: 'firm',     conf: 0.85 },
};

export function lookupGrip(label: string): { grip: Grip; force: Force; conf: number } {
  return MAP[label] ?? { grip: 'no_grasp', force: 'firm', conf: 0.9 };   // unbekannt → sichere Verweigerung
}
