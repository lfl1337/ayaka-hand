import type { TriggerSource } from '../types';
import { OnsetDetector, type OnsetConfig } from './onset';

/** Spielt eine EMG-Trace (rectified, [0,1]) mit sampleRateHz ab; tick(now) treibt sie.
 *  rng: austauschbare Rauschquelle (Tests: deterministisch), Default Math.random. */
export class ReplayTrigger implements TriggerSource {
  private cbs: ((tMs: number) => void)[] = [];
  private det: OnsetDetector;
  private running = false;
  private startedAt = 0;
  private lastIdx = -1;
  private sigma = 0;
  private trace: number[];
  private cfg: OnsetConfig;
  private rng: () => number;
  mav = 0;

  constructor(trace: number[], cfg: OnsetConfig, rng: () => number = Math.random) {
    this.trace = trace;
    this.cfg = cfg;
    this.rng = rng;
    this.det = new OnsetDetector(cfg);
  }

  start(): void { this.running = true; this.startedAt = -1; this.lastIdx = -1; }
  stop(): void { this.running = false; }
  onGo(cb: (tMs: number) => void): void { this.cbs.push(cb); }
  injectNoise(sigma: number): void { this.sigma = sigma; }

  /** Verarbeitet alle Samples zwischen letztem und aktuellem Zeitpunkt. */
  tick(nowMs: number): void {
    if (!this.running) return;
    if (this.startedAt < 0) this.startedAt = nowMs;
    const idx = Math.floor(((nowMs - this.startedAt) / 1000) * this.cfg.sampleRateHz);
    for (let i = this.lastIdx + 1; i <= idx; i++) {
      const base = this.trace[i % this.trace.length];
      const noisy = Math.max(0, base + this.sigma * (this.rng() * 2 - 1));
      const tSample = this.startedAt + (i / this.cfg.sampleRateHz) * 1000;
      this.mav = noisy;
      const go = this.det.push(noisy, tSample);
      if (go !== null) this.cbs.forEach((cb) => cb(go));
    }
    this.lastIdx = idx;
  }
}

/** Dev-Only: Space-Taste als GO. Nie in Demo-Aufnahmen verwenden. */
export class KeyboardTrigger implements TriggerSource {
  private cbs: ((tMs: number) => void)[] = [];
  private handler = (e: KeyboardEvent) => { if (e.code === 'Space') this.cbs.forEach((cb) => cb(performance.now())); };
  start(): void { window.addEventListener('keydown', this.handler); }
  stop(): void { window.removeEventListener('keydown', this.handler); }
  onGo(cb: (tMs: number) => void): void { this.cbs.push(cb); }
  injectNoise(): void { /* n/a */ }
}
