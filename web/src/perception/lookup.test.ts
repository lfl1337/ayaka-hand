// web/src/perception/lookup.test.ts
import { describe, expect, it } from 'vitest';
import { lookupGrip } from './lookup';

describe('lookupGrip (ablation strawman — CNN replaces this on day 4)', () => {
  it('maps everyday classes to sensible grips', () => {
    expect(lookupGrip('cup').grip).toBe('power');
    expect(lookupGrip('cell phone').grip).toBe('lateral');
    expect(lookupGrip('scissors').grip).toBe('lateral');
    expect(lookupGrip('apple').grip).toBe('tripod');
  });
  it('unknown classes refuse (no_grasp with high certainty)', () => {
    const u = lookupGrip('elephant');
    expect(u.grip).toBe('no_grasp');
    expect(u.conf).toBeGreaterThan(0.8);
  });
  it('fragile classes come back delicate', () => {
    expect(lookupGrip('wine glass').force).toBe('delicate');
  });
  it('open-vocabulary additions map correctly (LLMDet vocabulary)', () => {
    const egg = lookupGrip('egg');
    expect(egg.grip).toBe('tripod');
    expect(egg.force).toBe('delicate');        // Hazard-Beat: zerbrechlich → sanft
    expect(lookupGrip('key').grip).toBe('pinch');
    expect(lookupGrip('mug').grip).toBe('power');
  });
});
