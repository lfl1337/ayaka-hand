// web/src/perception/letterbox.ts
// Geometrie für Studio-Frames: /snap detektiert auf dem ORIGINAL-Foto (Boxen in absoluten
// Original-Pixeln, mitgeliefert als frame.size), broadcastet aber nur ein Preview-JPEG
// (längste Seite ≤640). Für die Box-Transformation ins Canvas MUSS darum frame.size die
// Referenz sein — naturalWidth/naturalHeight des dekodierten Previews sind nur der Fallback
// für Alt-Frames ohne size-Feld. (Preview erhält das Seitenverhältnis → drawImage streckt korrekt.)

export interface FrameGeometry {
  scale: number;   // Original-px → Canvas-px
  drawW: number;   // gezeichnete Bildbreite im Canvas
  drawH: number;
  offX: number;    // Letterbox-Offset
  offY: number;
}

export function studioFrameGeometry(
  size: [number, number] | undefined,
  naturalW: number,
  naturalH: number,
  canvasW: number,
  canvasH: number,
): FrameGeometry {
  const nw = size?.[0] || naturalW || canvasW;
  const nh = size?.[1] || naturalH || canvasH;
  const scale = nw > 0 && nh > 0 ? Math.min(canvasW / nw, canvasH / nh) : 1;
  const drawW = nw * scale;
  const drawH = nh * scale;
  return { scale, drawW, drawH, offX: (canvasW - drawW) / 2, offY: (canvasH - drawH) / 2 };
}
