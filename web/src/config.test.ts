import { describe, expect, it } from 'vitest';
import { CONFIG } from './config';
describe('config', () => {
  it('gates are ordered', () => {
    expect(CONFIG.gating.tauFull).toBeGreaterThan(CONFIG.gating.tauSoft);
    expect(CONFIG.onset.tHigh).toBeGreaterThan(CONFIG.onset.tLow);
  });
});
