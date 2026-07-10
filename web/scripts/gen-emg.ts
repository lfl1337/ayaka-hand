// web/scripts/gen-emg.ts — einmalig: pnpm tsx scripts/gen-emg.ts (oder node --experimental-strip-types)
// Erzeugt 8 s @200 Hz: Ruhe-Rauschen + 2 realistische Bursts (Anstieg/Plateau/Abfall).
import { writeFileSync } from 'node:fs';
const HZ = 200, SEC = 8, out: number[] = [];
const burst = (t: number, c: number, w: number) => Math.exp(-((t - c) ** 2) / (2 * w * w));
for (let i = 0; i < HZ * SEC; i++) {
  const t = i / HZ;
  const envelope = 0.9 * burst(t, 2.5, 0.25) + 0.85 * burst(t, 5.5, 0.3);
  const rest = 0.04 + 0.02 * Math.sin(t * 1.3);
  out.push(Number(Math.max(0, rest + envelope * (0.85 + 0.15 * Math.sin(t * 40))).toFixed(4)));
}
writeFileSync('public/replay/synthetic-emg.json', JSON.stringify({ sampleRateHz: HZ, label: 'synthetic demo trace (no subject data)', samples: out }));
console.log(`wrote ${out.length} samples`);
