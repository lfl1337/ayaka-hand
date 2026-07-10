import type { FsmEvent, FsmState, GraspHypothesis } from '../types';

export interface GatingConfig { tauFull: number; tauSoft: number; armedTimeoutMs: number }
export interface FsmOutput { state: FsmState; command: 'preshape' | 'close' | 'open' | null; note: string | null }

/** Soft-Lock-FSM. Invariante: JEDER Fehler-/Unsicherheits-Pfad endet mit command='open'. */
export class GraspFsm {
  private _state: FsmState = 'IDLE';
  private _hypothesis: GraspHypothesis | null = null;

  private cfg: GatingConfig;
  constructor(cfg: GatingConfig) { this.cfg = cfg; }

  get state(): FsmState { return this._state; }
  get hypothesis(): GraspHypothesis | null { return this._hypothesis; }

  dispatch(ev: FsmEvent, _tMs: number): FsmOutput {
    switch (ev.type) {
      case 'ERROR':
      case 'TIMEOUT':
        return this.toIdle('open', ev.type === 'ERROR' ? 'fail-safe-error' : 'armed-timeout');

      case 'HYPOTHESIS': {
        // GRASP hält: Vision-Zweifel darf ein gehaltenes (kamera-verdecktes) Objekt nie fallen lassen — nur RELEASE_CMD/ERROR/TIMEOUT öffnen.
        if (this._state === 'GRASP') return this.out(null, null);          // greifend: ignorieren
        const { h } = ev;
        if (h.grip === 'no_grasp' || h.confidence < this.cfg.tauSoft) {
          return this.toIdle('open', h.grip === 'no_grasp' ? 'no-grasp' : 'refusal-low-confidence');
        }
        this._hypothesis = h;                                              // Soft-Lock: jederzeit revidierbar
        if (h.confidence >= this.cfg.tauFull) { this._state = 'ARMED'; return this.out('preshape', null); }
        this._state = 'PRESHAPE';
        return this.out('preshape', 'confirm-required');
      }

      case 'GO': {
        if (this._state === 'ARMED' || this._state === 'PRESHAPE') {
          const note = this._state === 'PRESHAPE' ? 'confirmed-mid-confidence' : null;
          this._state = 'GRASP';
          return this.out('close', note);
        }
        return this.out(null, this._state === 'IDLE' ? 'go-without-target' : null);
      }

      case 'RELEASE_CMD': {
        if (this._state !== 'GRASP') return this.out(null, null);
        return this.toIdle('open', 'released');
      }
    }
  }

  private toIdle(command: 'open', note: string | null): FsmOutput {
    this._state = 'IDLE'; this._hypothesis = null;
    return this.out(command, note);
  }
  private out(command: FsmOutput['command'], note: string | null): FsmOutput {
    return { state: this._state, command, note };
  }
}
