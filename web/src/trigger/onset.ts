// web/src/trigger/onset.ts
export interface OnsetConfig {
  mavWindowMs: number; tHigh: number; tLow: number;
  holdMs: number; refractoryMs: number; sampleRateHz: number;
}

/** Double-Threshold-Onset auf gleitendem MAV. Kandidat bei tHigh; GO, wenn
 *  danach holdMs lang über tLow gehalten wird. Refraktärzeit gegen Doppel-GOs. */
export class OnsetDetector {
  private buf: number[] = [];
  private candidateAt: number | null = null;
  private lastGoAt = Number.NEGATIVE_INFINITY;
  private cfg: OnsetConfig;

  constructor(cfg: OnsetConfig) {
    this.cfg = cfg;
  }

  push(sample: number, tMs: number): number | null {
    const winN = Math.max(1, Math.round((this.cfg.mavWindowMs / 1000) * this.cfg.sampleRateHz));
    this.buf.push(Math.abs(sample));
    if (this.buf.length > winN) this.buf.shift();
    const mav = this.buf.reduce((a, b) => a + b, 0) / this.buf.length;

    if (mav < this.cfg.tLow) { this.candidateAt = null; return null; }   // abgefallen → reset
    if (mav >= this.cfg.tHigh && this.candidateAt === null) this.candidateAt = tMs;

    if (this.candidateAt !== null
      && tMs - this.candidateAt >= this.cfg.holdMs
      && tMs - this.lastGoAt >= this.cfg.refractoryMs) {
      this.lastGoAt = tMs;
      this.candidateAt = null;
      return tMs;                                                        // GO
    }
    return null;
  }
}
