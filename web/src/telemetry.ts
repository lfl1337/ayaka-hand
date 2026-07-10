// Opt-in live telemetry beacon. The network POST is off unless the page URL carries ?live=1;
// the LOCAL fan-out (onLocalEvent) runs always, so the in-page HUD event feed works offline too.
// Fire-and-forget verification tooling for the remote gate — never a product path,
// so every failure is swallowed: a down sink must never disturb the demo.
type Beam = (event: string, data?: Record<string, unknown>) => void;
type LocalListener = (event: string, data?: Record<string, unknown>) => void;

const localListeners: LocalListener[] = [];

/** In-Page-Abonnent für JEDES gebeamte Event (unabhängig vom ?live=1-Gate) — speist den HUD-Feed. */
export function onLocalEvent(cb: LocalListener): void {
  localListeners.push(cb);
}

function fanOut(event: string, data?: Record<string, unknown>): void {
  for (const cb of localListeners) {
    try { cb(event, data); } catch { /* ein kaputter Abonnent darf den Fluss nicht stoppen */ }
  }
}

export function initTelemetry(): Beam {
  const live = new URLSearchParams(location.search).get('live') === '1';
  const url = `http://${location.hostname}:4174/e`;
  const beam: Beam = (event, data) => {
    fanOut(event, data);                       // IMMER lokal spiegeln (Feed lebt ohne ?live=1)
    if (!live) return;                         // der Netzwerk-POST ist der opt-in-Teil
    try {
      // text/plain keeps it a CORS-simple request (no preflight); no-cors + keepalive
      // let it survive navigations. No queue, no retry — a missed beam is acceptable.
      fetch(url, {
        method: 'POST',
        mode: 'no-cors',
        keepalive: true,
        headers: { 'Content-Type': 'text/plain' },
        body: JSON.stringify({ t: Date.now(), event, ...data }),
      }).catch(() => {});
    } catch {
      // sink unreachable / fetch threw synchronously — ignore.
    }
  };
  if (live) {
    // Live-Gate-Sichtbarkeit: sonst unsichtbare JS-Fehler / abgelehnte Promises ans Sink melden,
    // damit ein stiller Stall (Handy: „nichts passiert") in der Telemetrie auftaucht statt zu verschwinden.
    window.addEventListener('error', (e) => beam('js-error', { msg: String(e.message).slice(0, 200) }));
    window.addEventListener('unhandledrejection', (e) => beam('promise-rejection', { msg: String(e.reason).slice(0, 200) }));
  }
  return beam;
}
