import type { Grip } from '../types';

const CYCLE: Grip[] = ['power', 'lateral', 'pinch', 'tripod'];

export class ManualMode {
  private i = 0;
  get current(): Grip { return CYCLE[this.i]; }
  next(): Grip { this.i = (this.i + 1) % CYCLE.length; return this.current; }
}
