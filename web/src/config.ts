export const CONFIG = {
  gating: { tauFull: 0.75, tauSoft: 0.5, armedTimeoutMs: 15000 },       // lokaler WASM (D-FINE bzw. rtdetr-Fallback): COCO-Closed-Set-Scores 0.7–0.97
  gatingOvd: { tauFull: 0.5, tauSoft: 0.35, armedTimeoutMs: 15000 },    // OVD (LLMDet/grounding): Scores liegen systematisch tiefer (~0.46–0.66) → eigene Kalibrierung
  voting: { window: 7, tauHigh: 0.6, tauLow: 0.4 },
  onset: { mavWindowMs: 100, tHigh: 0.5, tLow: 0.2, holdMs: 80, refractoryMs: 500, sampleRateHz: 200 },
  hand: { preshapeMs: 180, closeMs: 350, releaseMs: 600, delicateCurl: 0.7 },
  detector: { scoreThreshold: 0.4, modelPath: `${import.meta.env.BASE_URL}models/dfine/` },      // D-FINE-Small (Apache-2.0); Swap = nur der Pfad — rtdetr bleibt als Fallback unter /models/rtdetr/ auf Platte
} as const;
