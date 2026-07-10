/** Scrollender EMG-Strip: MAV-Signal (eine Serie) + beide Schwellenlinien + GO-Marker.
 *  Farben kommen aus den Viz-Tokens (CSS Custom Properties), einmal gelesen und bei Theme-Wechsel
 *  neu — kein Hardcode. Eine Serie → keine Legende (die Caption unter dem Canvas benennt sie). */
interface StripColors { signal: string; threshold: string; go: string; muted: string }

export class StripChart {
  private buf: number[] = [];
  private goMarks: number[] = [];
  private ctx: CanvasRenderingContext2D;
  private tHigh: number;
  private tLow: number;
  private capacity: number;
  private colors: StripColors = { signal: '#2a78d6', threshold: '#eda100', go: '#008300', muted: '#5c6570' };

  constructor(ctx: CanvasRenderingContext2D, tHigh: number, tLow: number, capacity = 600) {
    this.ctx = ctx;
    this.tHigh = tHigh;
    this.tLow = tLow;
    this.capacity = capacity;
    this.refreshColors();
    // Tokens wechseln zwischen Light/Dark → bei Theme-Wechsel neu einlesen.
    if (typeof matchMedia === 'function') {
      matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.refreshColors());
    }
  }

  /** Viz-Tokens aus dem CSS lesen (mit Hardcode-Fallback fürs Headless-Rendering). */
  refreshColors(): void {
    if (typeof document === 'undefined') return;
    const cs = getComputedStyle(document.body);
    const read = (name: string, fb: string): string => cs.getPropertyValue(name).trim() || fb;
    this.colors = {
      signal: read('--viz-signal', '#2a78d6'),
      threshold: read('--viz-threshold', '#eda100'),
      go: read('--viz-go', '#008300'),
      muted: read('--color-text-muted', '#5c6570'),
    };
  }

  push(v: number): void { this.buf.push(v); if (this.buf.length > this.capacity) { this.buf.shift(); this.goMarks = this.goMarks.map((g) => g - 1).filter((g) => g >= 0); } }
  markGo(): void { this.goMarks.push(this.buf.length - 1); }

  draw(): void {
    const ctx = this.ctx;
    const { width: w, height: h } = ctx.canvas;
    ctx.clearRect(0, 0, w, h);
    const y = (v: number): number => h - v * h * 0.9 - 2;

    // Schwellen: --viz-threshold gestrichelt (hairline) + winzige rechtsbündige Labels in Text-Muted.
    const threshold = (thr: number, dash: number[], label: string): void => {
      ctx.setLineDash(dash);
      ctx.strokeStyle = this.colors.threshold; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, y(thr)); ctx.lineTo(w, y(thr)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = this.colors.muted; ctx.font = '10px system-ui, sans-serif';
      ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText(label, w - 3, y(thr) - 1);
    };
    threshold(this.tHigh, [6, 4], 'τ¹');
    threshold(this.tLow, [2, 4], 'τ²');

    // Signal: eine Serie, --viz-signal 2px, runde Joins.
    ctx.strokeStyle = this.colors.signal; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.beginPath();
    this.buf.forEach((v, i) => { const x = (i / this.capacity) * w; i === 0 ? ctx.moveTo(x, y(v)) : ctx.lineTo(x, y(v)); });
    ctx.stroke();

    // GO-Marker: --viz-go 2px vertikal + kleines ▲GO am unteren Rand (Status-Mark mit Label, nie Farbe allein).
    ctx.fillStyle = this.colors.go; ctx.font = '9px system-ui, sans-serif'; ctx.textBaseline = 'bottom';
    for (const g of this.goMarks) {
      const x = (g / this.capacity) * w;
      ctx.fillRect(x - 1, 0, 2, h);
      const nearRight = x > w - 20;
      ctx.textAlign = nearRight ? 'right' : 'left';
      ctx.fillText('▲GO', nearRight ? x - 2 : x + 2, h - 1);
    }
  }
}
