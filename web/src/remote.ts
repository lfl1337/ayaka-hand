// web/src/remote.ts
// Opt-in studio remote mode. The hosted judge demo stays pure-local (WASM); studio
// inference is enabled ONLY via the ?remote= URL param, never by default.
const DEFAULT_PORT = 27461;

/** Resolve the studio inference endpoint from the URL (no trailing slash).
 *  ?remote=1 | ?remote=default → http://<host>:27461 (the SSH-tunnelled Mac server)
 *  ?remote=http(s)://…         → that URL verbatim
 *  absent / unrecognized       → null (stay on the local WASM path — fail-safe). */
export function getRemoteEndpoint(): string | null {
  const raw = new URLSearchParams(location.search).get('remote');
  if (!raw) return null;
  if (raw === '1' || raw === 'default') return `http://${location.hostname}:${DEFAULT_PORT}`;
  if (/^https?:\/\//i.test(raw)) return raw.replace(/\/+$/, ''); // trim trailing slash so `${endpoint}/infer` stays clean
  return null;
}
