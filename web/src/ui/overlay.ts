// web/src/ui/overlay.ts
import type { Detection } from '../perception/ranking';

type Box = { xmin: number; ymin: number; xmax: number; ymax: number };

/** CSS-Custom-Property live lesen (Overlay reagiert so auf Light/Dark), mit Fallback fürs Testen. */
function cssVar(name: string, fallback: string): string {
  if (typeof document === 'undefined') return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function boxPath(ctx: CanvasRenderingContext2D, b: Box): void {
  ctx.beginPath();
  ctx.rect(b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin);
}

/** roundRect ohne Abhängigkeit von der (uneinheitlich getypten) Canvas-API — via arcTo. */
function roundRectPath(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/** Tracking-Overlay auf das #snap-Canvas — IMMER nach dem Bild aufrufen.
 *  Nicht-gewählte Boxen: Halo (2.5px dunkel #0008 unter 1.5px hell #fffc) → auf jedem Foto lesbar.
 *  Gewählte Box: 3px in --viz-signal + Label-Chip (Klasse + Score) oben-links.
 *  Kontaktpunkt (normiert auf die gewählte Box): 8px-Punkt in --viz-go mit 2px Surface-Ring. */
export function drawOverlay(
  ctx: CanvasRenderingContext2D,
  ranked: Detection[],
  chosenIdx: number,
  contactPoint?: { x: number; y: number },
  cortexLabel?: string,
): void {
  const surface = cssVar('--color-surface', '#f5f4f1');
  const textCol = cssVar('--color-text', '#1d232a');
  const signal = cssVar('--viz-signal', '#2a78d6');
  const go = cssVar('--viz-go', '#008300');
  const primary = cssVar('--color-primary', '#2f6f8f');

  ctx.save();
  ctx.lineJoin = 'round';

  ranked.forEach((d, i) => {
    if (i === chosenIdx) return;                             // gewählte Box zuletzt (liegt oben)
    boxPath(ctx, d.box); ctx.lineWidth = 2.5; ctx.strokeStyle = '#0008'; ctx.stroke();
    boxPath(ctx, d.box); ctx.lineWidth = 1.5; ctx.strokeStyle = '#fffc'; ctx.stroke();
  });

  const chosen = ranked[chosenIdx] as Detection | undefined;
  if (chosen) {
    const b = chosen.box;
    boxPath(ctx, b); ctx.lineWidth = 3; ctx.strokeStyle = signal; ctx.stroke();

    // Label-Chip: Surface-Hintergrund, 4px Radius, Text in Textfarbe, 12px — oben-links an der Box.
    ctx.font = '12px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    const label = `${chosen.label} ${chosen.score.toFixed(2)}`;
    const padX = 6, chipH = 18;
    const chipW = ctx.measureText(label).width + padX * 2;
    const chipX = Math.max(0, Math.min(b.xmin, ctx.canvas.width - chipW));   // im Canvas halten
    const chipY = Math.max(0, b.ymin - chipH);                               // über der Oberkante, nie oben abgeschnitten
    roundRectPath(ctx, chipX, chipY, chipW, chipH, 4);
    ctx.fillStyle = surface; ctx.fill();
    ctx.fillStyle = textCol;
    ctx.fillText(label, chipX + padX, chipY + chipH / 2);

    // Cortex-Override am Objekt sichtbar: zweiter Chip UNTER der Box, sobald Cortex ein anderes Label sieht.
    if (cortexLabel && cortexLabel !== chosen.label) {
      const cText = `Edge: ${chosen.label} → Cortex: ${cortexLabel}`;
      const cW = ctx.measureText(cText).width + padX * 2;
      const cX = Math.max(0, Math.min(b.xmin, ctx.canvas.width - cW));       // im Canvas halten
      const cY = Math.min(ctx.canvas.height - chipH, b.ymax);                // unter der Box, nie unten abgeschnitten
      roundRectPath(ctx, cX, cY, cW, chipH, 4);
      ctx.fillStyle = primary; ctx.fill();                                   // --color-primary = Cortex-Akzent (wie im Panel)
      ctx.fillStyle = '#fff';
      ctx.fillText(cText, cX + padX, cY + chipH / 2);
    }

    if (contactPoint) {
      const px = b.xmin + contactPoint.x * (b.xmax - b.xmin);
      const py = b.ymin + contactPoint.y * (b.ymax - b.ymin);
      ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = go; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = surface; ctx.stroke();       // Ring hält den Punkt auf jedem Untergrund lesbar
    }
  }
  ctx.restore();
}
