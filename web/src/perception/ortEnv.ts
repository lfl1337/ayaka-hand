// web/src/perception/ortEnv.ts
// EINE Quelle für die ORT-WASM-Runtime-Wahl. Detektor (transformers.js) und Student
// (onnxruntime-web) teilen sich denselben Backend-Init — wer zuerst lädt, setzt die Pfade.
// Beide MÜSSEN exakt dieselbe /ort/-Datei wählen (Safari plain vs. asyncify), sonst lädt
// der zweitinitialisierende Modul einen inkompatiblen Build. Vorher lebte diese Logik
// verbatim in detector.ts UND student.ts — genau die Drift, vor der ihre Kommentare warnten.

/** Spiegelbild von transformers.js' isSafari(): der WebGPU-Build lädt zur Laufzeit auf
 *  Safari/WebKit den plain-, sonst den asyncify-WASM. iOS-Fremdbrowser (CriOS & Co.) laufen
 *  bewusst auf dem asyncify-Zweig — wie transformers.js selbst. */
export function isAppleWebkit(): boolean {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  const isAppleVendor = (navigator.vendor || '').indexOf('Apple') > -1;
  const notOtherBrowser = !/CriOS|FxiOS|EdgiOS|OPiOS|mercury|brave/i.test(ua)
    && !ua.includes('Chrome') && !ua.includes('Android');
  return isAppleVendor && notOtherBrowser;
}

const ORT_BASE = '/ort/';

/** Exaktes self-hosted Datei-Paar für diese Browser-Klasse (kein CDN beim Judge).
 *  Als {mjs,wasm}-Objekt, damit transformers.js den Binary vorab same-origin cached. */
export function ortWasmPaths(): { mjs: string; wasm: string } {
  return isAppleWebkit()
    ? { mjs: `${ORT_BASE}ort-wasm-simd-threaded.mjs`, wasm: `${ORT_BASE}ort-wasm-simd-threaded.wasm` }
    : { mjs: `${ORT_BASE}ort-wasm-simd-threaded.asyncify.mjs`, wasm: `${ORT_BASE}ort-wasm-simd-threaded.asyncify.wasm` };
}

/** Single-Thread-Baseline: ohne crossOriginIsolation (plain HTTP/statisches Hosting)
 *  gibt es kein SharedArrayBuffer. */
export const ORT_NUM_THREADS = 1;
