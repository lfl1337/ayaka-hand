// web/src/perception/ranking.test.ts
import { describe, expect, it } from 'vitest';
import { rankDetections, type Detection } from './ranking';

const det = (label: string, score: number, cx: number, cy: number, s: number): Detection =>
  ({ label, score, box: { xmin: cx - s / 2, ymin: cy - s / 2, xmax: cx + s / 2, ymax: cy + s / 2 } });

describe('rankDetections ("reach is the pointer")', () => {
  it('prefers the centered object over an off-center one of equal score', () => {
    const r = rankDetections([det('cup', 0.8, 320, 240, 100), det('bottle', 0.8, 60, 60, 100)], 640, 480);
    expect(r[0].label).toBe('cup');
  });
  it('bigger (closer) object wins between equally centered ones', () => {
    const r = rankDetections([det('cup', 0.8, 320, 240, 60), det('bowl', 0.8, 320, 240, 160)], 640, 480);
    expect(r[0].label).toBe('bowl');
  });
  it('empty input stays empty', () => {
    expect(rankDetections([], 640, 480)).toEqual([]);
  });
});
