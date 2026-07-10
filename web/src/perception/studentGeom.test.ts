import { describe, expect, it } from 'vitest';
import { cropPlan, softmax } from './studentGeom';

describe('cropPlan', () => {
  it('scales the shorter side to 176 and centers the 160 crop', () => {
    // 200x100 box: shorter side 100 -> scale 1.76 -> 352x176; center crop 160
    const p = cropPlan({ xmin: 0, ymin: 0, xmax: 200, ymax: 100 }, 640, 480);
    expect(p.dh).toBeCloseTo(176, 3);
    expect(p.dw).toBeCloseTo(352, 3);
    expect(p.dx).toBeCloseTo((160 - 352) / 2, 3);   // horizontal überstand, zentriert
    expect(p.dy).toBeCloseTo((160 - 176) / 2, 3);
  });

  it('clamps the box to canvas bounds', () => {
    const p = cropPlan({ xmin: -20, ymin: -10, xmax: 100, ymax: 90 }, 640, 480);
    expect(p.sx).toBe(0); expect(p.sy).toBe(0);
    expect(p.sw).toBeCloseTo(100, 3); expect(p.sh).toBeCloseTo(90, 3);
  });

  it('survives degenerate boxes (min 1px source)', () => {
    const p = cropPlan({ xmin: 50, ymin: 50, xmax: 50, ymax: 50 }, 640, 480);
    expect(p.sw).toBeGreaterThanOrEqual(1);
    expect(p.sh).toBeGreaterThanOrEqual(1);
    expect(Number.isFinite(p.dw)).toBe(true);
  });
});

describe('softmax', () => {
  it('sums to 1 and orders correctly', () => {
    const s = softmax([2, 1, 0.1]);
    expect(s.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(s[0]).toBeGreaterThan(s[1]);
  });
  it('is numerically stable for large logits', () => {
    const s = softmax([1000, 999]);
    expect(Number.isFinite(s[0])).toBe(true);
    expect(s[0]).toBeGreaterThan(s[1]);
  });
});
