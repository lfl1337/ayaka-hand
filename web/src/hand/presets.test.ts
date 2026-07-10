import { describe, expect, it } from 'vitest';
import { GRIP_PRESETS, closedCurls } from './presets';
import { ThreeHand } from './threeHand';

const CFG = { preshapeMs: 180, closeMs: 350, releaseMs: 600, delicateCurl: 0.7 };

describe('presets', () => {
  it('defines all five grips with curls in [0,1]', () => {
    for (const g of ['power', 'lateral', 'pinch', 'tripod', 'no_grasp'] as const) {
      const p = GRIP_PRESETS[g];
      for (const v of Object.values(p)) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1); }
    }
  });
  it('pinch and power are visibly distinct preshapes', () => {
    expect(Math.abs(GRIP_PRESETS.pinch.ring - GRIP_PRESETS.power.ring)).toBeGreaterThan(0.2);
  });
  it('delicate close stops short of firm close', () => {
    const d = closedCurls('pinch', 'delicate', CFG.delicateCurl);
    const f = closedCurls('pinch', 'firm', CFG.delicateCurl);
    expect(d.index).toBeLessThan(f.index);
  });
});

describe('ThreeHand motion (headless)', () => {
  it('reaches the preshape within preshapeMs and reports state', () => {
    const h = new ThreeHand(CFG);
    h.preshape({ grip: 'tripod', force: 'firm', confidence: 1, source: 'edge' });
    for (let t = 0; t < 200; t += 16) h.tick(16);
    expect(h.state).toBe('preshaped');
    expect(Math.abs(h.curls.index - GRIP_PRESETS.tripod.index)).toBeLessThan(0.05);
  });
  it('open() always wins immediately (fail-safe)', () => {
    const h = new ThreeHand(CFG);
    h.preshape({ grip: 'power', force: 'firm', confidence: 1, source: 'edge' });
    h.close('firm');
    h.open();
    for (let t = 0; t < 700; t += 16) h.tick(16);
    expect(h.state).toBe('open');
    expect(h.curls.index).toBeLessThanOrEqual(0.05);
  });
});
