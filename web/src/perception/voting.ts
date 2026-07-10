// web/src/perception/voting.ts
import type { Grip } from '../types';

export interface VoteConfig { window: number; tauHigh: number; tauLow: number }

export class TemporalVoter {
  private buf: { grip: Grip; conf: number }[] = [];
  private current: Grip = 'no_grasp';
  private cfg: VoteConfig;

  constructor(cfg: VoteConfig) {
    this.cfg = cfg;
  }

  push(grip: Grip, conf: number): { grip: Grip; conf: number } {
    this.buf.push({ grip, conf });
    if (this.buf.length > this.cfg.window) this.buf.shift();

    const tally = new Map<Grip, { n: number; sum: number }>();
    for (const e of this.buf) {
      const t = tally.get(e.grip) ?? { n: 0, sum: 0 };
      t.n += 1; t.sum += e.conf; tally.set(e.grip, t);
    }
    let best: Grip = 'no_grasp', bestN = 0, bestMean = 0;
    for (const [g, t] of tally) {
      const mean = t.sum / t.n;
      if (t.n > bestN || (t.n === bestN && mean > bestMean)) { best = g; bestN = t.n; bestMean = mean; }
    }
    const eff = (bestN / this.buf.length) * bestMean;   // Anteil × mittlere Confidence

    if (this.current === best) {
      if (eff < this.cfg.tauLow) this.current = 'no_grasp';           // Schmitt: Release
    } else if (eff >= this.cfg.tauHigh) {
      this.current = best;                                            // Schmitt: Acquire
    } else if (this.current !== 'no_grasp' && best !== this.current) {
      const cur = tally.get(this.current);
      const curEff = cur ? (cur.n / this.buf.length) * (cur.sum / cur.n) : 0;
      if (curEff < this.cfg.tauLow) this.current = 'no_grasp';        // alter Griff kollabiert
    }
    return { grip: this.current, conf: eff };
  }

  reset(): void { this.buf = []; this.current = 'no_grasp'; }
}
