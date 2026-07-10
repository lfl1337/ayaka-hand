import { describe, expect, it } from 'vitest';
import { extractResult } from './detectRemote';

const det = { label: 'cup', score: 0.52, box: { xmin: 0, ymin: 0, xmax: 10, ymax: 10 } };

describe('extractResult', () => {
  it('passes the server detector backend through — OVD scores need OVD gates', () => {
    const r = extractResult({ detections: [det], detector: 'ovd', ms: 230, size: [4032, 3024] });
    expect(r.detector).toBe('ovd');
    expect(r.detections).toHaveLength(1);
  });

  it('accepts rtdetr and drops unknown detector values', () => {
    expect(extractResult({ detections: [], detector: 'rtdetr' }).detector).toBe('rtdetr');
    expect(extractResult({ detections: [], detector: 'llmdet_base (ovd)' }).detector).toBeUndefined();
  });

  it('keeps backward-compat with bare arrays and envelopes without detector', () => {
    expect(extractResult([det])).toEqual({ detections: [det] });
    expect(extractResult({ detections: [det] }).detector).toBeUndefined();
    expect(extractResult({ nope: true }).detections).toEqual([]);
  });
});
