import { describe, expect, it } from 'vitest';
import { qrTargetUrl } from './qrShare';

describe('qrTargetUrl', () => {
  it('liest href vom übergebenen Location-Objekt', () => {
    expect(qrTargetUrl({ href: 'http://localhost:4173/' })).toBe('http://localhost:4173/');
  });

  it('spiegelt Query-Parameter wie ?remote=1 wider', () => {
    expect(qrTargetUrl({ href: 'http://localhost:4173/?remote=1' })).toBe('http://localhost:4173/?remote=1');
  });
});
