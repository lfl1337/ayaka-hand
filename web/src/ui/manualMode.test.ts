import { describe, expect, it } from 'vitest';
import { ManualMode } from './manualMode';

describe('ManualMode', () => {
  it('cycles through the four grips', () => {
    const m = new ManualMode();
    expect(m.current).toBe('power');
    expect(m.next()).toBe('lateral');
    expect(m.next()).toBe('pinch');
    expect(m.next()).toBe('tripod');
    expect(m.next()).toBe('power');
  });
});
