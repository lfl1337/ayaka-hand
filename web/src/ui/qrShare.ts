// web/src/ui/qrShare.ts
import * as QRCode from 'qrcode';

/** Aktuelle Seiten-URL für den Share-QR. `location`-Parameter statt direktem window.location-Zugriff --
 *  diese Codebase testet ohne DOM (vitest.config.ts: environment 'node', kein jsdom) und der Default
 *  wird nur im echten Browser ausgewertet (Default-Parameter sind lazy, evaluieren nur ohne Argument). */
export function qrTargetUrl(location: { href: string } = window.location): string {
  return location.href;
}

/** Togglet den QR-Popover per Button-Klick, rendert den QR frisch bei jedem Öffnen (nie gecacht --
 *  URL kann sich durch Modus-Query-Params ändern). Schließen: erneuter Klick, Klick außerhalb, Escape.
 *  DOM-Wiring wie renderStates/renderFeed in hud.ts -- bewusst ungetestet (jsdom kann Canvas nicht
 *  sinnvoll rendern), gleiches Muster wie overlay.ts in dieser Codebase. */
export function initQrShare(button: HTMLButtonElement, popover: HTMLElement, canvas: HTMLCanvasElement): void {
  const urlEl = popover.querySelector<HTMLElement>('#qr-url');

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
    const url = qrTargetUrl();
    if (urlEl) urlEl.textContent = url;
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
