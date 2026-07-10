export type Grip = 'power' | 'lateral' | 'pinch' | 'tripod' | 'no_grasp';
export type Force = 'delicate' | 'firm';

export interface GraspHypothesis {
  grip: Grip;
  force: Force;
  confidence: number;                       // ∈ [0,1]
  source: 'edge' | 'cortex';
  objectLabel?: string;
  contactRegion?: string;
  contactPoint?: { x: number; y: number };
  hazards?: string[];
  rationale?: string;
  via?: 'cnn' | 'lookup';                   // Griff-Quelle im Reflex-Pfad: destilliertes CNN vs. Label-Lookup (Fail-safe)
}

export interface TriggerSource {
  start(): void;
  stop(): void;
  onGo(cb: (tMs: number) => void): void;
  injectNoise(sigma: number): void;
}

export type HandPose = 'open' | 'preshaped' | 'closed';

export interface Hand {
  preshape(h: GraspHypothesis): void;
  close(force: Force): void;
  open(): void;
  readonly state: HandPose;
}

export type FsmState = 'IDLE' | 'PRESHAPE' | 'ARMED' | 'GRASP' | 'RELEASE';

export type FsmEvent =
  | { type: 'HYPOTHESIS'; h: GraspHypothesis }
  | { type: 'GO'; tMs: number }
  | { type: 'RELEASE_CMD' }
  | { type: 'TIMEOUT' }
  | { type: 'ERROR' };
