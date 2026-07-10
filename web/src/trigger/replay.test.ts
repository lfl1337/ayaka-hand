import { describe, expect, it } from 'vitest';
import { ReplayTrigger } from './replay';

const CFG = { mavWindowMs: 25, tHigh: 0.5, tLow: 0.2, holdMs: 80, refractoryMs: 500, sampleRateHz: 200 };
// Trace: 0.5 s Ruhe, 0.5 s Burst — bei 200 Hz
const trace = [...Array(100).fill(0.03), ...Array(100).fill(0.85)];

describe('ReplayTrigger', () => {
  it('emits GO during the burst when noise is off', () => {
    const r = new ReplayTrigger(trace, CFG);
    const gos: number[] = [];
    r.onGo((t) => gos.push(t));
    r.start();
    for (let t = 0; t <= 1000; t += 5) r.tick(t);
    expect(gos.length).toBe(1);
    expect(gos[0]).toBeGreaterThanOrEqual(500 + CFG.holdMs);
  });
  it('loops the trace and stays silent when stopped', () => {
    const r = new ReplayTrigger(trace, CFG);
    const gos: number[] = [];
    r.onGo((t) => gos.push(t));
    r.start();
    for (let t = 0; t <= 2100; t += 5) r.tick(t);   // ~2 Loops → 2 GOs
    expect(gos.length).toBe(2);
    r.stop();
    for (let t = 2100; t <= 3000; t += 5) r.tick(t);
    expect(gos.length).toBe(2);
  });
  it('deterministic noise does not fire during rest', () => {
    const r = new ReplayTrigger(Array(400).fill(0.03), CFG, () => 0.99); // fixe "Zufalls"-Quelle
    const gos: number[] = [];
    r.onGo((t) => gos.push(t));
    r.injectNoise(0.08);
    r.start();
    for (let t = 0; t <= 2000; t += 5) r.tick(t);
    expect(gos.length).toBe(0); // 0.03 + 0.08*0.99 << tHigh — Double-Threshold hält
  });
});
