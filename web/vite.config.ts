import { defineConfig } from 'vite';

// appType 'mpa': this app has no client-side router (index.html + a standalone
// capture.html, mode-switching is done via query params on the same page) — the
// SPA default silently serves index.html (200, text/html) for ANY unmatched path.
// transformers.js probes optional per-model files (generation_config.json,
// tokenizer_config.json) that legitimately don't exist for a vision-only model;
// with the SPA fallback on, those probes get back HTML instead of a 404 and
// JSON.parse throws ("Unexpected token '<'"), killing the whole detector load.
export default defineConfig({
  appType: 'mpa',
});
