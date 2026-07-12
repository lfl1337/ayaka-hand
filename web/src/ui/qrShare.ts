// web/src/ui/qrShare.ts
import * as QRCode from 'qrcode';
import { getRemoteEndpoint } from '../remote';

type Loc = { href: string; origin: string; pathname: string };

/** Ziel-URL für den Share-QR.
 *  Studio-Modus (?remote=): der QR zeigt auf die leichtgewichtige `capture.html`. Das Handy wird
 *  damit zur KAMERA, die per POST an /snap in die laufende Desktop-Session einspeist (Server → SSE),
 *  statt eine zweite, eigenständige Demo zu öffnen. `?to=` trägt den Server DIESES Desktops, damit
 *  das Handy ihn erreicht. Ohne Studio-Server gibt es keine Session zum Andocken → der QR öffnet die
 *  Demo eigenständig auf dem Handy (dessen Rückkamera).
 *  loc + remote sind injizierbar, weil diese Codebase ohne DOM testet (vitest environment 'node'). */
export function qrTargetUrl(loc: Loc = window.location, remote: string | null = null): string {
  if (remote) {
    const dir = loc.pathname.replace(/[^/]*$/, '');            // Verzeichnis der aktuellen Seite (z.B. /ayaka-hand/)
    return `${loc.origin}${dir}capture.html?to=${encodeURIComponent(remote)}`;
  }
  return loc.href;
}

/** Togglet den QR-Popover per Button-Klick, rendert den QR frisch bei jedem Öffnen (nie gecacht --
 *  Ziel-URL hängt vom Studio-Modus ab). Schließen: erneuter Klick, Klick außerhalb, Escape.
 *  DOM-Wiring wie renderStates/renderFeed in hud.ts -- bewusst ungetestet (jsdom kann Canvas nicht
 *  sinnvoll rendern), gleiches Muster wie overlay.ts in dieser Codebase. */
export function initQrShare(button: HTMLButtonElement, popover: HTMLElement, canvas: HTMLCanvasElement): void {
  const urlEl = popover.querySelector<HTMLElement>('#qr-url');
  const captionEl = popover.querySelector<HTMLElement>('#qr-caption');

  function close(): void {
    popover.hidden = true;
    document.removeEventListener('click', onOutsideClick);
    document.removeEventListener('keydown', onKeydown);
  }

  function onOutsideClick(e: MouseEvent): void {
    if (!popover.contains(e.target as Node) && e.target !== button) close();
  }

  function onKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') close();
  }

  async function open(): Promise<void> {
    const remote = getRemoteEndpoint();
    const url = qrTargetUrl(window.location, remote);
    if (urlEl) urlEl.textContent = url;
    if (captionEl) {
      captionEl.textContent = remote
        ? 'Scan to use your phone as the camera'      // Studio: Handy → capture.html → Server → diese Session
        : 'Scan to open the demo on your phone';       // kein Server: eigenständige Demo (Rückkamera)
    }
    popover.hidden = false;
    try {
      await QRCode.toCanvas(canvas, url, { width: 200, margin: 1 });
    } catch (err) {
      console.error('qr-share', err);                    // URL bleibt als Text nutzbar, Canvas bleibt leer
    }
    document.addEventListener('click', onOutsideClick);
    document.addEventListener('keydown', onKeydown);
  }

  button.addEventListener('click', (e) => {
    e.stopPropagation();
    if (popover.hidden) void open(); else close();
  });
}
