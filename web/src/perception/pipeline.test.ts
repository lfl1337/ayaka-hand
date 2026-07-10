import { describe, expect, it, vi } from 'vitest';
import type { Detection } from './ranking';
import { hypothesisFromDetections } from './pipeline';

vi.mock('./student', () => ({
  studentReady: vi.fn(() => true),
  inferGrip: vi.fn(async () => ({ grip: 'pinch', force: 'delicate', gripProb: 0.8 })),
  loadStudent: vi.fn(async () => undefined),
}));

const det = (label: string, score: number): Detection => ({
  label, score, box: { xmin: 280, ymin: 200, xmax: 360, ymax: 280 }, // zentriert im 640×480
});

describe('hypothesisFromDetections', () => {
  it('leere Detektionen → fail-safe no_grasp (conf 0.95)', () => {
    expect(hypothesisFromDetections([], 640, 480)).toEqual({
      grip: 'no_grasp', force: 'firm', confidence: 0.95, source: 'edge', objectLabel: 'nothing detected',
    });
  });

  it('bekanntes Label → Lookup-Griff, conf = score × mapConf, contactPoint mittig, source edge', () => {
    const h = hypothesisFromDetections([det('cup', 0.8)], 640, 480);
    expect(h.grip).toBe('power');
    expect(h.force).toBe('firm');
    expect(h.confidence).toBeCloseTo(0.72, 5);   // 0.8 × 0.9
    expect(h.source).toBe('edge');
    expect(h.objectLabel).toBe('cup');
    expect(h.contactPoint).toEqual({ x: 0.5, y: 0.5 });
  });

  it('confidence wird auf 1 geklemmt', () => {
    expect(hypothesisFromDetections([det('cup', 2)], 640, 480).confidence).toBe(1);
  });

  it('unbekanntes Label → no_grasp aus dem Lookup', () => {
    expect(hypothesisFromDetections([det('giraffe', 0.9)], 640, 480).grip).toBe('no_grasp');
  });
});

describe('analyzeDetailedWithStudent', () => {
  it('analyzeDetailedWithStudent overrides grip/force from the CNN', async () => {
    const { analyzeDetailedWithStudent } = await import('./pipeline');
    const dets = [{ label: 'cup', score: 0.9, box: { xmin: 0, ymin: 0, xmax: 50, ymax: 50 } }];
    const res = await analyzeDetailedWithStudent(dets, 640, 480, {} as HTMLCanvasElement);
    expect(res.hypothesis.grip).toBe('pinch');
    expect(res.hypothesis.force).toBe('delicate');
    expect(res.hypothesis.via).toBe('cnn');
    expect(res.hypothesis.confidence).toBeCloseTo(Math.min(1, 0.9 * 0.8), 6);
    expect(res.hypothesis.objectLabel).toBe('cup');
  });

  it('falls back to lookup when the student returns null', async () => {
    const student = await import('./student');
    (student.inferGrip as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);
    const { analyzeDetailedWithStudent } = await import('./pipeline');
    const dets = [{ label: 'cup', score: 0.9, box: { xmin: 0, ymin: 0, xmax: 50, ymax: 50 } }];
    const res = await analyzeDetailedWithStudent(dets, 640, 480, {} as HTMLCanvasElement);
    expect(res.hypothesis.grip).toBe('power');           // lookup('cup')
    expect(res.hypothesis.via).toBe('lookup');
  });

  it('no_grasp/empty ranking never calls the student', async () => {
    const student = await import('./student');
    const spy = student.inferGrip as ReturnType<typeof vi.fn>;
    spy.mockClear();
    const { analyzeDetailedWithStudent } = await import('./pipeline');
    const res = await analyzeDetailedWithStudent([], 640, 480, {} as HTMLCanvasElement);
    expect(res.hypothesis.grip).toBe('no_grasp');
    expect(spy).not.toHaveBeenCalled();
  });

  it('studentReady false → lookup unverändert, via lookup, Student wird nie befragt', async () => {
    const student = await import('./student');
    (student.studentReady as ReturnType<typeof vi.fn>).mockReturnValueOnce(false);
    const spy = student.inferGrip as ReturnType<typeof vi.fn>;
    spy.mockClear();
    const { analyzeDetailedWithStudent } = await import('./pipeline');
    const dets = [{ label: 'cup', score: 0.9, box: { xmin: 0, ymin: 0, xmax: 50, ymax: 50 } }];
    const res = await analyzeDetailedWithStudent(dets, 640, 480, {} as HTMLCanvasElement);
    expect(res.hypothesis.grip).toBe('power');           // lookup('cup') unverändert
    expect(res.hypothesis.force).toBe('firm');
    expect(res.hypothesis.via).toBe('lookup');
    expect(spy).not.toHaveBeenCalled();
  });

  it('CNN-confidence wird auf 1 geklemmt (score × gripProb > 1)', async () => {
    const student = await import('./student');
    (student.inferGrip as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ grip: 'pinch', force: 'delicate', gripProb: 1.5 });
    const { analyzeDetailedWithStudent } = await import('./pipeline');
    const dets = [{ label: 'cup', score: 0.9, box: { xmin: 0, ymin: 0, xmax: 50, ymax: 50 } }];
    const res = await analyzeDetailedWithStudent(dets, 640, 480, {} as HTMLCanvasElement);
    expect(res.hypothesis.confidence).toBe(1);           // 0.9 × 1.5 = 1.35 → geklemmt
    expect(res.hypothesis.via).toBe('cnn');
  });
});
