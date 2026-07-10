export type EvalEventType = 'snapshot' | 'control_action' | 'mode_switch' | 'go' | 'grasp' | 'release' | 'refusal';

export class EvalLogger {
  private events: { type: EvalEventType; tMs: number }[] = [];
  mark(type: EvalEventType, tMs: number): void { this.events.push({ type, tMs }); }
  summary() {
    const first = (t: EvalEventType) => this.events.find((e) => e.type === t)?.tMs ?? null;
    const count = (t: EvalEventType) => this.events.filter((e) => e.type === t).length;
    const snap = first('snapshot'), grasp = first('grasp');
    return {
      controlActions: count('control_action'),
      modeSwitches: count('mode_switch'),
      refusals: count('refusal'),
      timeToGraspMs: snap !== null && grasp !== null ? grasp - snap : null,
      events: [...this.events],
    };
  }
  toJson(): string { return JSON.stringify(this.summary(), null, 1); }
  reset(): void { this.events = []; }
}
