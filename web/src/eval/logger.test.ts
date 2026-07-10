import { describe, expect, it } from 'vitest';
import { EvalLogger } from './logger';

describe('EvalLogger', () => {
  it('counts control actions and computes time-to-grasp', () => {
    const l = new EvalLogger();
    l.mark('snapshot', 1000);
    l.mark('control_action', 1000);
    l.mark('go', 2500);
    l.mark('grasp', 2600);
    const s = l.summary();
    expect(s.controlActions).toBe(1);
    expect(s.timeToGraspMs).toBe(1600);   // snapshot → grasp
    expect(s.events.length).toBe(4);
  });
  it('mode switches are counted separately', () => {
    const l = new EvalLogger();
    l.mark('mode_switch', 0); l.mark('mode_switch', 10);
    expect(l.summary().modeSwitches).toBe(2);
  });
});
