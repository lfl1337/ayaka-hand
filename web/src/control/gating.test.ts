import { describe, expect, it } from 'vitest';
import { applyGatesForDetector } from './gating';
import { CONFIG } from '../config';

describe('applyGatesForDetector', () => {
  it("detector='ovd' schreibt die tiefere OVD-Kalibrierung in target", () => {
    const g = { tauFull: 0.75, tauSoft: 0.5, armedTimeoutMs: 15000 };
    applyGatesForDetector(g, 'ovd');
    expect(g).toEqual({ tauFull: 0.5, tauSoft: 0.35, armedTimeoutMs: 15000 });
  });
  it("detector='rtdetr' schreibt die Default-Gates (zurück)", () => {
    const g = { tauFull: 0.5, tauSoft: 0.35, armedTimeoutMs: 15000 };
    applyGatesForDetector(g, 'rtdetr');
    expect(g).toEqual({ tauFull: 0.75, tauSoft: 0.5, armedTimeoutMs: 15000 });
  });
  it('mutiert dasselbe Objekt (FSM-Referenz überlebt) und lässt die CONFIG-Presets unberührt', () => {
    const g = { tauFull: 0.75, tauSoft: 0.5, armedTimeoutMs: 15000 };
    const ref = g;
    applyGatesForDetector(g, 'ovd');
    expect(ref).toBe(g);                          // gleiche Referenz, in-place mutiert
    expect(CONFIG.gatingOvd.tauFull).toBe(0.5);   // Preset nicht überschrieben
    expect(CONFIG.gating.tauFull).toBe(0.75);
  });
});
