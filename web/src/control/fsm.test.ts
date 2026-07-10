import { describe, expect, it } from 'vitest';
import { GraspFsm } from './fsm';
import type { GraspHypothesis } from '../types';

const CFG = { tauFull: 0.75, tauSoft: 0.5, armedTimeoutMs: 15000 };
const hyp = (grip: GraspHypothesis['grip'], confidence: number): GraspHypothesis =>
  ({ grip, force: 'firm', confidence, source: 'edge' });

describe('GraspFsm', () => {
  it('happy path: high conf → ARMED → GO → GRASP → RELEASE → IDLE', () => {
    const f = new GraspFsm(CFG);
    expect(f.dispatch({ type: 'HYPOTHESIS', h: hyp('power', 0.9) }, 0).command).toBe('preshape');
    expect(f.state).toBe('ARMED');
    expect(f.dispatch({ type: 'GO', tMs: 100 }, 100).command).toBe('close');
    expect(f.state).toBe('GRASP');
    expect(f.dispatch({ type: 'RELEASE_CMD' }, 2000).command).toBe('open');
    expect(f.state).toBe('IDLE');
  });
  it('refusal: low confidence keeps hand open, GO is ignored in IDLE', () => {
    const f = new GraspFsm(CFG);
    const out = f.dispatch({ type: 'HYPOTHESIS', h: hyp('power', 0.3) }, 0);
    expect(out.command).toBe('open');
    expect(f.state).toBe('IDLE');
    expect(f.dispatch({ type: 'GO', tMs: 50 }, 50).command).toBe(null);
  });
  it('mid band: PRESHAPE with confirm note; GO confirms', () => {
    const f = new GraspFsm(CFG);
    const out = f.dispatch({ type: 'HYPOTHESIS', h: hyp('pinch', 0.6) }, 0);
    expect(out.command).toBe('preshape');
    expect(f.state).toBe('PRESHAPE');
    expect(out.note).toBe('confirm-required');
    expect(f.dispatch({ type: 'GO', tMs: 80 }, 80).command).toBe('close');
    expect(f.state).toBe('GRASP');
  });
  it('soft-lock: ARMED revises on a new hypothesis (reversible)', () => {
    const f = new GraspFsm(CFG);
    f.dispatch({ type: 'HYPOTHESIS', h: hyp('power', 0.9) }, 0);
    const out = f.dispatch({ type: 'HYPOTHESIS', h: hyp('tripod', 0.85) }, 30);
    expect(f.state).toBe('ARMED');
    expect(out.command).toBe('preshape');
    expect(f.hypothesis?.grip).toBe('tripod');
  });
  it('no_grasp or ERROR from any state fails safe to OPEN', () => {
    const f = new GraspFsm(CFG);
    f.dispatch({ type: 'HYPOTHESIS', h: hyp('power', 0.9) }, 0);
    expect(f.dispatch({ type: 'HYPOTHESIS', h: hyp('no_grasp', 0.9) }, 10).command).toBe('open');
    expect(f.state).toBe('IDLE');
    f.dispatch({ type: 'HYPOTHESIS', h: hyp('power', 0.9) }, 20);
    f.dispatch({ type: 'GO', tMs: 40 }, 40);
    expect(f.dispatch({ type: 'ERROR' }, 50).command).toBe('open');
    expect(f.state).toBe('IDLE');
  });
  it('TIMEOUT in ARMED opens the hand', () => {
    const f = new GraspFsm(CFG);
    f.dispatch({ type: 'HYPOTHESIS', h: hyp('power', 0.9) }, 0);
    expect(f.dispatch({ type: 'TIMEOUT' }, 16000).command).toBe('open');
    expect(f.state).toBe('IDLE');
  });
  it('GO during GRASP is ignored (fsm-level refractory)', () => {
    const f = new GraspFsm(CFG);
    f.dispatch({ type: 'HYPOTHESIS', h: hyp('power', 0.9) }, 0);
    f.dispatch({ type: 'GO', tMs: 100 }, 100);
    expect(f.dispatch({ type: 'GO', tMs: 150 }, 150).command).toBe(null);
    expect(f.state).toBe('GRASP');
  });
  it('grasp-hold: vision doubt during GRASP never opens (held object occludes camera)', () => {
    const f = new GraspFsm(CFG);
    f.dispatch({ type: 'HYPOTHESIS', h: hyp('power', 0.9) }, 0);
    f.dispatch({ type: 'GO', tMs: 100 }, 100);
    expect(f.state).toBe('GRASP');
    // no_grasp while holding: GRASP guard short-circuits before the no_grasp/refusal check.
    const noGrasp = f.dispatch({ type: 'HYPOTHESIS', h: hyp('no_grasp', 0.9) }, 200);
    expect(noGrasp.command).toBe(null);
    expect(f.state).toBe('GRASP');
    // near-zero confidence while holding: same — drop-safety dominates, hand must not open.
    const lowConf = f.dispatch({ type: 'HYPOTHESIS', h: hyp('power', 0.1) }, 300);
    expect(lowConf.command).toBe(null);
    expect(f.state).toBe('GRASP');
    expect(f.hypothesis?.grip).toBe('power'); // held target not clobbered by doubt
  });
});
