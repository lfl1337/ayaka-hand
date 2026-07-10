// web/src/perception/ranking.ts
export interface Detection {
  label: string; score: number;
  box: { xmin: number; ymin: number; xmax: number; ymax: number };
}

/** Snapshot-Variante von „reach is the pointer": Score × Zentrums-Nähe × Größe.
 *  (Looming-Rate braucht Video — Phase-2/Video-Modus.) */
export function rankDetections(dets: Detection[], imgW: number, imgH: number): Detection[] {
  const cx = imgW / 2, cy = imgH / 2, diag = Math.hypot(imgW, imgH) / 2;
  const value = (d: Detection) => {
    const bx = (d.box.xmin + d.box.xmax) / 2, by = (d.box.ymin + d.box.ymax) / 2;
    const centerness = 1 - Math.hypot(bx - cx, by - cy) / diag;
    const area = Math.max(0, d.box.xmax - d.box.xmin) * Math.max(0, d.box.ymax - d.box.ymin);
    const size = Math.sqrt(area / (imgW * imgH));
    return d.score * (0.5 + 0.5 * centerness) * (0.5 + 0.5 * Math.min(1, size * 3));
  };
  return [...dets].sort((a, b) => value(b) - value(a));
}
