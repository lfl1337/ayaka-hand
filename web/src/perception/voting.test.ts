// web/src/perception/voting.test.ts
import { describe, expect, it } from 'vitest';
import { TemporalVoter } from './voting';

const mk = () => new TemporalVoter({ window: 5, tauHigh: 0.6, tauLow: 0.4 });

describe('TemporalVoter', () => {
  it('starts at no_grasp and acquires after consistent votes', () => {
    const v = mk();
    expect(v.push('power', 0.9).grip).toBe('power'); // share=1, mean=0.9 → eff=0.9 ≥ tauHigh
    v.reset();
    // low confidence never acquires
    for (let i = 0; i < 5; i++) expect(v.push('power', 0.3).grip).toBe('no_grasp');
  });
  it('holds current grip in hysteresis band (soft-lock friendly)', () => {
    const v = mk();
    for (let i = 0; i < 5; i++) v.push('power', 0.9);
    expect(v.push('pinch', 0.9).grip).toBe('power'); // 1 Ausreißer kippt nicht
    expect(v.push('pinch', 0.9).grip).toBe('power'); // eff(power)=3/5*0.9=0.54 > tauLow
  });
  it('switches when the new grip clearly wins', () => {
    const v = mk();
    for (let i = 0; i < 5; i++) v.push('power', 0.9);
    for (let i = 0; i < 5; i++) v.push('pinch', 0.9);
    expect(v.push('pinch', 0.9).grip).toBe('pinch');
  });
  it('releases to no_grasp when confidence collapses', () => {
    const v = mk();
    for (let i = 0; i < 5; i++) v.push('power', 0.9);
    let out = { grip: 'power', conf: 1 };
    for (let i = 0; i < 5; i++) out = v.push('power', 0.3);
    expect(out.grip).toBe('no_grasp'); // eff=0.3 < tauLow
  });
});
