import { describe, expect, it } from 'vitest';
import { meterModel, pushEvent } from './hud';

describe('meterModel', () => {
  it('unter τ_soft → refuse, pct = conf×100', () => {
    expect(meterModel(0.3, 0.5, 0.75)).toEqual({ pct: 30, zone: 'refuse' });
  });
  it('zwischen τ_soft und τ_full → confirm', () => {
    expect(meterModel(0.6, 0.5, 0.75)).toEqual({ pct: 60, zone: 'confirm' });
  });
  it('ab τ_full → full, pct auf 100 geklemmt', () => {
    expect(meterModel(1.4, 0.5, 0.75)).toEqual({ pct: 100, zone: 'full' });
  });
  it('OVD-Gates (τ_soft 0.35 / τ_full 0.5): 0.52 ≥ τ_full → full', () => {
    expect(meterModel(0.52, 0.35, 0.5)).toEqual({ pct: 52, zone: 'full' });
  });
  it('OVD-Gates: 0.46 im Confirm-Band [0.35, 0.5)', () => {
    expect(meterModel(0.46, 0.35, 0.5)).toEqual({ pct: 46, zone: 'confirm' });
  });
});

describe('pushEvent', () => {
  it('hängt neueste zuerst an, ohne den Eingabepuffer zu mutieren', () => {
    const buf = ['b', 'a'];
    expect(pushEvent(buf, 'c', 8)).toEqual(['c', 'b', 'a']);
    expect(buf).toEqual(['b', 'a']);                                  // pur: Original unverändert
  });
  it('kappt auf cap und wirft das älteste raus', () => {
    const full = ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'];           // 8 Einträge
    const out = pushEvent(full, 'i', 8);
    expect(out).toHaveLength(8);
    expect(out[0]).toBe('i');                                         // neu vorne
    expect(out).not.toContain('a');                                  // ältestes verdrängt
  });
});
