import { describe, expect, it } from 'vitest';
import { OnsetDetector } from './onset';

const CFG = { mavWindowMs: 25, tHigh: 0.5, tLow: 0.2, holdMs: 80, refractoryMs: 500, sampleRateHz: 200 };
const DT = 1000 / CFG.sampleRateHz; // 5 ms

function feed(d: OnsetDetector, samples: number[], t0 = 0): number[] {
  const gos: number[] = [];
  samples.forEach((s, i) => { const g = d.push(s, t0 + i * DT); if (g !== null) gos.push(g); });
  return gos;
}

describe('OnsetDetector', () => {
  it('fires exactly once on a sustained burst, after the hold time', () => {
    const d = new OnsetDetector(CFG);
    const gos = feed(d, [...Array(40).fill(0.05), ...Array(60).fill(0.9)]);
    expect(gos.length).toBe(1);
    expect(gos[0]).toBeGreaterThanOrEqual(40 * DT + CFG.holdMs); // frühestens Kandidat+Hold
    expect(gos[0]).toBeLessThan(40 * DT + CFG.holdMs + 10 * DT); // und nicht viel später
  });
  it('rejects a spike shorter than the hold time', () => {
    const d = new OnsetDetector(CFG);
    const gos = feed(d, [...Array(40).fill(0.05), ...Array(8).fill(0.9), ...Array(60).fill(0.05)]); // 40ms Burst
    expect(gos.length).toBe(0);
  });
  it('respects the refractory period', () => {
    const d = new OnsetDetector(CFG);
    const burst = [...Array(30).fill(0.9), ...Array(10).fill(0.0)];
    const gos = feed(d, [...burst, ...burst, ...Array(120).fill(0.0), ...burst]);
    expect(gos.length).toBe(2); // Burst 2 fällt in die Refraktärzeit von Burst 1
  });
});
