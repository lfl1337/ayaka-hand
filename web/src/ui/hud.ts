// web/src/ui/hud.ts
import type { FsmState } from '../types';

// ---------- pure helpers (unit-tested) ----------

export type MeterZone = 'refuse' | 'confirm' | 'full';
export interface MeterModel { pct: number; zone: MeterZone; }

/** Konfidenz → Balkenmodell. pct 0..100 (geklemmt); Zone durch die beiden Gating-Schwellen:
 *  < tauSoft = refuse · [tauSoft, tauFull) = confirm · >= tauFull = full. Pur → testbar. */
export function meterModel(conf: number, tauSoft: number, tauFull: number): MeterModel {
  const pct = Math.max(0, Math.min(100, conf * 100));
  const zone: MeterZone = conf < tauSoft ? 'refuse' : conf < tauFull ? 'confirm' : 'full';
  return { pct, zone };
}

/** Ringpuffer, neueste zuerst, gekappt auf cap. Pur (mutiert `buf` nicht) → testbar. */
export function pushEvent<T>(buf: readonly T[], e: T, cap = 8): T[] {
  return [e, ...buf].slice(0, cap);
}

// ---------- DOM renderers ----------

const STATES: readonly FsmState[] = ['IDLE', 'PRESHAPE', 'ARMED', 'GRASP', 'RELEASE'];

/** Zustands-Timeline: aktiver Chip = fett + Rahmen + führender Punkt (●); Rest gedämpfter Text. */
export function renderStates(el: HTMLElement, active: FsmState): void {
  el.replaceChildren(...STATES.map((s) => {
    const chip = document.createElement('span');
    chip.className = s === active ? 'hud-chip is-active' : 'hud-chip';
    if (s === active) {
      const dot = document.createElement('span');
      dot.className = 'hud-dot'; dot.textContent = '●';
      chip.append(dot, document.createTextNode(s));
    } else {
      chip.textContent = s;
    }
    return chip;
  }));
}

/** Konfidenzmeter: Balken 0..1, Füllung --viz-signal auf Track --viz-muted, Ticks bei tauSoft/tauFull
 *  mit winzigen Labels, aktueller Wert als Text daneben. */
export function renderConfMeter(el: HTMLElement, conf: number, tauSoft: number, tauFull: number): void {
  const m = meterModel(conf, tauSoft, tauFull);
  const track = document.createElement('div');
  track.className = 'hud-meter-track';
  const fill = document.createElement('div');
  fill.className = 'hud-meter-fill';
  fill.style.width = `${m.pct}%`;
  track.appendChild(fill);
  for (const tau of [tauSoft, tauFull]) {
    const tick = document.createElement('span');
    tick.className = 'hud-meter-tick';
    tick.style.left = `${tau * 100}%`;
    const lab = document.createElement('span');
    lab.className = 'hud-meter-ticklabel';
    lab.textContent = tau.toFixed(2);
    tick.appendChild(lab);
    track.appendChild(tick);
  }
  const val = document.createElement('span');
  val.className = 'hud-meter-val';
  val.textContent = conf.toFixed(2);
  el.replaceChildren(track, val);
}

export interface TileModel {
  detectorMs: number | null;
  reflexMs: number | null;
  grip: string;
}

/** Drei Stat-Tiles (Hero-Zahlen, Text-Tokens, keine Farbcodierung). */
export function renderTiles(el: HTMLElement, m: TileModel): void {
  const tiles: readonly [string, string, string][] = [
    ['Detektor', fmtMs(m.detectorMs), 'ms'],
    ['Reflex GO→Griff', fmtMs(m.reflexMs), 'ms'],
    ['Griff', m.grip || '–', ''],
  ];
  el.replaceChildren(...tiles.map(([label, value, unit]) => {
    const tile = document.createElement('div');
    tile.className = 'hud-tile';
    const l = document.createElement('div'); l.className = 'hud-tile-label'; l.textContent = label;
    const v = document.createElement('div'); v.className = 'hud-tile-value'; v.textContent = value;
    tile.append(l, v);
    if (unit) {
      const u = document.createElement('span'); u.className = 'hud-tile-unit'; u.textContent = unit;
      v.appendChild(u);
    }
    return tile;
  }));
}

function fmtMs(ms: number | null): string {
  return ms == null ? '–' : String(Math.round(ms));
}

export interface FeedItem { t: number; text: string; }

/** Event-Feed: neueste zuerst, monospace, gedämpfter Text. */
export function renderFeed(el: HTMLElement, items: readonly FeedItem[]): void {
  el.replaceChildren(...items.map((it) => {
    const row = document.createElement('div');
    row.className = 'hud-feed-row';
    row.textContent = `${clock(it.t)}  ${it.text}`;
    return row;
  }));
}

function clock(t: number): string {
  const d = new Date(t);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
function pad(n: number): string { return n < 10 ? `0${n}` : String(n); }
