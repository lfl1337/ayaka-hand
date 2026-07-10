import { describe, expect, it } from 'vitest';
import { studioFrameGeometry } from './letterbox';

describe('studioFrameGeometry', () => {
  // Der Studio-Kontrakt: /snap liefert Boxen in ORIGINAL-Pixeln (frame.size), aber das
  // Preview-JPEG ist auf längste Seite ≤640 verkleinert → naturalWidth des <img> ist der
  // FALSCHE Raum für die Box-Transformation. frame.size muss gewinnen.
  it('uses frame.size over the (downscaled) preview naturalWidth', () => {
    // Handy-Foto 4032×3024, Preview 640×480, Canvas 640×480
    const g = studioFrameGeometry([4032, 3024], 640, 480, 640, 480);
    expect(g.scale).toBeCloseTo(640 / 4032, 6);
    // Box-Mitte des Originals landet in der Canvas-Mitte
    expect(2016 * g.scale + g.offX).toBeCloseTo(320, 3);
    expect(1512 * g.scale + g.offY).toBeCloseTo(240, 3);
  });

  it('falls back to natural dims when frame.size is missing (Alt-Frames)', () => {
    const g = studioFrameGeometry(undefined, 640, 480, 640, 480);
    expect(g.scale).toBe(1);
    expect(g.offX).toBe(0);
    expect(g.offY).toBe(0);
  });

  it('letterboxes portrait photos horizontally centered', () => {
    // Hochformat 3024×4032 in 640×480 → Höhe limitiert, seitliche Balken
    const g = studioFrameGeometry([3024, 4032], 480, 640, 640, 480);
    expect(g.scale).toBeCloseTo(480 / 4032, 6);
    expect(g.drawH).toBeCloseTo(480, 3);
    expect(g.offX).toBeGreaterThan(0);
    expect(g.offY).toBeCloseTo(0, 3);
  });

  it('never divides by zero on degenerate input', () => {
    const g = studioFrameGeometry([0, 0], 0, 0, 640, 480);
    expect(Number.isFinite(g.scale)).toBe(true);
    expect(g.scale).toBe(1);
  });
});
