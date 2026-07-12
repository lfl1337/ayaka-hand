import { describe, expect, it } from 'vitest';
import { qrTargetUrl } from './qrShare';

const loc = (href: string, origin: string, pathname: string) => ({ href, origin, pathname });

describe('qrTargetUrl', () => {
  it('ohne Studio-Server: gibt die aktuelle Seiten-URL zurück (Demo öffnet eigenständig)', () => {
    expect(qrTargetUrl(loc('http://localhost:4173/', 'http://localhost:4173', '/'))).toBe('http://localhost:4173/');
  });

  it('ohne Studio-Server: spiegelt vorhandene Query-Parameter', () => {
    const l = loc('http://localhost:4173/?x=1', 'http://localhost:4173', '/');
    expect(qrTargetUrl(l)).toBe('http://localhost:4173/?x=1');
  });

  it('Studio-Modus: zeigt auf capture.html mit dem Server als ?to (Handy = Kamera)', () => {
    const l = loc('http://192.168.1.5:4173/ayaka-hand/?remote=1', 'http://192.168.1.5:4173', '/ayaka-hand/');
    expect(qrTargetUrl(l, 'http://192.168.1.5:27461')).toBe(
      'http://192.168.1.5:4173/ayaka-hand/capture.html?to=http%3A%2F%2F192.168.1.5%3A27461');
  });

  it('Studio-Modus: Verzeichnis der Seite bleibt erhalten, index.html wird ersetzt', () => {
    const l = loc('http://host:4173/index.html?remote=1', 'http://host:4173', '/index.html');
    expect(qrTargetUrl(l, 'http://host:27461')).toBe(
      'http://host:4173/capture.html?to=http%3A%2F%2Fhost%3A27461');
  });
});
