import type { Force, Grip } from '../types';

export type FingerCurls = { thumb: number; index: number; middle: number; ring: number; pinky: number; spread: number };

/** Preshape-Aperturen (NICHT geschlossen): Hand öffnet sich passend zum Griff. */
export const GRIP_PRESETS: Record<Grip, FingerCurls> = {
  power:    { thumb: 0.25, index: 0.30, middle: 0.30, ring: 0.30, pinky: 0.30, spread: 0.15 },
  lateral:  { thumb: 0.15, index: 0.55, middle: 0.60, ring: 0.65, pinky: 0.65, spread: 0.05 },
  pinch:    { thumb: 0.30, index: 0.35, middle: 0.75, ring: 0.85, pinky: 0.85, spread: 0.10 },
  tripod:   { thumb: 0.30, index: 0.35, middle: 0.35, ring: 0.85, pinky: 0.85, spread: 0.12 },
  no_grasp: { thumb: 0.05, index: 0.05, middle: 0.05, ring: 0.05, pinky: 0.05, spread: 0.25 },
};

/** Ziel-Curls beim Schließen; delicate stoppt bei delicateCurl-Anteil des Weges (2-Stufen-Kraft). */
export function closedCurls(grip: Grip, force: Force, delicateCurl: number): FingerCurls {
  const closed: Record<Grip, FingerCurls> = {
    power:    { thumb: 0.85, index: 0.95, middle: 0.95, ring: 0.95, pinky: 0.95, spread: 0.05 },
    lateral:  { thumb: 0.75, index: 0.70, middle: 0.80, ring: 0.85, pinky: 0.85, spread: 0.02 },
    pinch:    { thumb: 0.70, index: 0.75, middle: 0.80, ring: 0.90, pinky: 0.90, spread: 0.08 },
    tripod:   { thumb: 0.70, index: 0.75, middle: 0.75, ring: 0.90, pinky: 0.90, spread: 0.10 },
    no_grasp: GRIP_PRESETS.no_grasp,
  };
  const base = GRIP_PRESETS[grip], target = closed[grip];
  if (force === 'firm') return target;
  const mix = (a: number, b: number) => a + (b - a) * delicateCurl;
  return { thumb: mix(base.thumb, target.thumb), index: mix(base.index, target.index),
           middle: mix(base.middle, target.middle), ring: mix(base.ring, target.ring),
           pinky: mix(base.pinky, target.pinky), spread: mix(base.spread, target.spread) };
}
