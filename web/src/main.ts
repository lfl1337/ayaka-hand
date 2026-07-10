import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import './style.css';
import { CONFIG } from './config';
import { GraspFsm, type GatingConfig } from './control/fsm';
import { type DetectorKind, applyGatesForDetector } from './control/gating';
import { EvalLogger } from './eval/logger';
import { ThreeHand } from './hand/threeHand';
import { analyzeDetailedWithStudent, analyzeSnapshot } from './perception/pipeline';
import { studioFrameGeometry } from './perception/letterbox';
import { detectorStatus, setDetectorProgressListener } from './perception/detector';
import { loadStudent } from './perception/student';
import type { Detection } from './perception/ranking';
import type { Force, GraspHypothesis, Grip } from './types';
import { getRemoteEndpoint } from './remote';
import { ReplayTrigger } from './trigger/replay';
import { ManualMode } from './ui/manualMode';
import { StripChart } from './ui/stripchart';
import { drawOverlay } from './ui/overlay';
import { type FeedItem, type TileModel, pushEvent, renderConfMeter, renderFeed, renderStates, renderTiles } from './ui/hud';
import { initTelemetry, onLocalEvent } from './telemetry';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// Aktive Gate-Kalibrierung: eine mutierbare Kopie, die die FSM per Referenz hält. Pro Frame auf den
// Detektor umgestellt (OVD ⇄ rtdetr) — mutiert, nie neu gebaut, damit FSM-Zustand + Hypothese überleben.
const activeGating: GatingConfig = { ...CONFIG.gating };
const fsm = new GraspFsm(activeGating);
const hand = new ThreeHand(CONFIG.hand);
const log = new EvalLogger();
const beam = initTelemetry();                 // Netzwerk-POST opt-in (?live=1); lokaler Fan-out läuft immer

// --- HUD (Pane 4 · Status) — Feed abonniert JEDES gebeamte Event, auch ohne ?live=1 ---
const hudStatesEl = $('hud-states');
const hudConfEl = $('hud-conf');
const hudTilesEl = $('hud-tiles');
const hudFeedEl = $('hud-feed');
const tiles: TileModel = { detectorMs: null, reflexMs: null, grip: '–' };
let feed: FeedItem[] = [];
renderStates(hudStatesEl, 'IDLE');
$('fsm-state').dataset.tone = 'muted';                            // Boot-Zustand neutral, bis die FSM erstmals dispatcht
renderConfMeter(hudConfEl, 0, activeGating.tauSoft, activeGating.tauFull);
renderTiles(hudTilesEl, tiles);
renderFeed(hudFeedEl, feed);
onLocalEvent((event, data) => {
  feed = pushEvent(feed, { t: Date.now(), text: feedText(event, data) });
  renderFeed(hudFeedEl, feed);
});
function feedText(event: string, data?: Record<string, unknown>): string {
  if (!data) return event;
  const bits = Object.entries(data).slice(0, 2).map(([k, v]) => `${k}=${v}`).join(' ');
  return bits ? `${event} ${bits}` : event;
}
/** Meter + Griff-Tile aus einer Hypothese aktualisieren (jeder Pfad, der eine Hypothese landet). */
function onHypothesis(h: GraspHypothesis): void {
  renderConfMeter(hudConfEl, h.confidence, activeGating.tauSoft, activeGating.tauFull);
  tiles.grip = h.grip === 'no_grasp' ? 'no_grasp' : `${h.grip} · ${h.force}`;
  renderTiles(hudTilesEl, tiles);
}
beam('page-load');

// --- UI-Referenzen + veränderlicher Zustand ---
const video = $('cam') as HTMLVideoElement;
const hypothesisEl = $('hypothesis');
hypothesisEl.dataset.tone = 'muted';           // Boot: kein Ergebnis → Ghost-Chip (untoned Badge reißt AA im Dark)
const btnSnapshot = $('btn-snapshot') as HTMLButtonElement;
const filePick = $('file-pick') as HTMLInputElement;
const btnCycle = $('btn-cycle') as HTMLButtonElement;
let lastHypothesisAt: number | null = null;   // Binding #2: Startpunkt für den timer-losen ARMED-TIMEOUT
let busy = false;                              // Snapshot läuft → Doppelklick/Concurrency sperren
let manualOn = false;
let lastBeamedPct = -10;                       // Progress-Beams auf jede 10-%-Stufe drosseln
let reflexT0: number | null = null;            // GO→Griff-Messung: bei GO gesetzt, direkt nach hand.close verrechnet

// --- Three.js Viewport (an Container gemessen → mobil skalierbar) ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 4 / 3, 0.1, 50);
camera.position.set(0, 1.4, 2.15);                               // etwas näher (fov 38) → Hand füllt ~70 % der Höhe
// Dreipunkt-Licht: Key wirft weiche Schatten (PCFSoft), Fill = Ambient, Rim dim von hinten.
const key = new THREE.DirectionalLight(0xffffff, 1.25);
key.position.set(2, 3.2, 2.2); key.castShadow = true;
key.shadow.mapSize.set(1024, 1024);
key.shadow.camera.near = 0.5; key.shadow.camera.far = 12;
key.shadow.camera.left = -3; key.shadow.camera.right = 3; key.shadow.camera.top = 3; key.shadow.camera.bottom = -3;
key.shadow.bias = -0.0005;                                       // gegen Shadow-Acne auf den Fingern
const rim = new THREE.DirectionalLight(0xbcd4e6, 0.5);
rim.position.set(-1.6, 1.4, -2.6);
scene.add(new THREE.AmbientLight(0xffffff, 0.25), key, rim);     // Ambient nur als Rest-Fill — die Env übernimmt
const viewport = $('hand-viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;              // filmische Kurve statt Clipping auf der Clearcoat-Shell
renderer.toneMappingExposure = 1.05;
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;   // PBR-Reflexe ohne HDR-Asset
pmrem.dispose();                                                 // Einmal-Werkzeug: interne Render-Targets/Shader sofort freigeben (Textur bleibt gültig)
viewport.appendChild(renderer.domElement);
hand.attachTo(scene);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enablePan = false;                                      // nur Rotieren + Zoom
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.autoRotate = true; controls.autoRotateSpeed = 0.7;      // Idle-Bühnendreher — erste Nutzer-Interaktion stoppt ihn
controls.addEventListener('start', () => { controls.autoRotate = false; });
controls.target.set(0, 0, -0.15); controls.update();
function sizeRenderer(): void {
  const w = viewport.clientWidth || 480;                          // an den Pane gemessen; Fallback falls Layout noch 0
  const h = w * 0.75;                                             // 4:3, konsistent zur Kamera-Aspect
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
sizeRenderer();
let resizeTimer = 0;
window.addEventListener('resize', () => {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(sizeRenderer, 150);             // 150 ms Debounce
});

// Detektor-Ladefehler menschenlesbar + auf ~80 Zeichen gekürzt für die Badge (voller Text geht per Telemetrie raus).
const detectorErrorText = (msg?: string): string =>
  msg ? `Detektor-Fehler: ${msg.slice(0, 80)}` : 'Detektor-Fehler — nochmal versuchen';

// --- Detektor-Ladefortschritt → Badge + Telemetrie (Sichtbarkeit statt Stille) ---
setDetectorProgressListener(({ state, pct, msg }) => {
  if (state === 'loading') {
    hypothesisEl.textContent = pct !== undefined ? `Modell lädt… ${pct} %` : 'Modell lädt…';
    if (pct !== undefined && pct >= lastBeamedPct + 10) { beam('detector', { state, pct }); lastBeamedPct = pct; }
  } else if (state === 'ready') {
    lastBeamedPct = -10;
    beam('detector', { state });
    if (busy) hypothesisEl.textContent = 'analysiere…';           // Modell fertig, Snapshot läuft noch weiter
  } else if (state === 'failed') {
    hypothesisEl.textContent = detectorErrorText(msg);
    beam('detector-error', { msg });                              // reale Ladefehler-Meldung ans Sink — sonst im Live-Gate unsichtbar (Fehler wird intern gefangen)
  }
});

// --- Destillierter 2,2M-Student laden. Der Detektor lädt LAZY (erst beim ersten Snapshot), also kann der
// Student den geteilten onnxruntime-web-Backend zuerst initialisieren — deshalb wählt student.ts dieselbe
// exakte /ort/-Datei wie detector.ts (sonst lädt der zweite Init einen inkompatiblen WASM-Build).
// Erfolg: der Reflex-Pfad ersetzt den Label-Lookup durch das CNN. Fehler: Badge bleibt auf Lookup-Baseline,
// analyzeDetailedWithStudent fällt via studentReady()===false ohnehin auf den Lookup zurück (die FSM merkt nichts).
const edgeBadge = $('edge-badge');
edgeBadge.dataset.tone = 'muted';                                 // Ladephase neutral; ok/warn erst nach dem Ergebnis
const studentWatchdog = window.setTimeout(() => {                 // hängender ONNX-Fetch (Promise settelt nie) → Badge darf nicht ewig „lädt…" zeigen;
  edgeBadge.textContent = '2,2-Mio-Reflex: Lookup-Baseline (Student lädt noch…)';   // der Lookup-Reflexpfad läuft derweil längst (studentReady()-Fallback)
  edgeBadge.dataset.tone = 'warn';
  beam('student-load-timeout');
}, 15_000);
void loadStudent().then(
  () => { window.clearTimeout(studentWatchdog); edgeBadge.textContent = '2,2-Mio-Reflex: CNN aktiv'; edgeBadge.dataset.tone = 'ok'; },
  (err: unknown) => {
    window.clearTimeout(studentWatchdog);
    edgeBadge.textContent = '2,2-Mio-Reflex: Lookup-Baseline (Fallback)';
    edgeBadge.dataset.tone = 'warn';
    beam('student-load-error', { msg: String((err as { message?: unknown })?.message ?? err) });
  },
);
// Hand-Modell-Fehler (threeHand meldet per CustomEvent): sichtbar machen statt nur Konsole —
// sonst steht beim Judging eine leere Bühne ohne jedes Signal, warum.
window.addEventListener('ayaka:hand-model-error', (e) => {
  const msg = String((e as CustomEvent).detail ?? 'unbekannt');
  beam('hand-model-error', { msg });
  const fsmBadge = $('fsm-state');
  fsmBadge.textContent = `Hand-Modell fehlt: ${msg.slice(0, 60)}`;
  fsmBadge.dataset.tone = 'warn';
});

// --- Trigger + Chart ---
const chart = new StripChart(($('emg') as HTMLCanvasElement).getContext('2d')!, CONFIG.onset.tHigh, CONFIG.onset.tLow);
let trigger: ReplayTrigger | undefined;
fetch(`${import.meta.env.BASE_URL}replay/synthetic-emg.json`).then((r) => r.json()).then((j) => {
  trigger = new ReplayTrigger(j.samples, CONFIG.onset);
  trigger.onGo((tMs) => {
    reflexT0 = performance.now();                                // Start der GO→Griff-Messung (Ende: direkt nach hand.close)
    log.mark('go', tMs); chart.markGo(); beam('go');
    apply(fsm.dispatch({ type: 'GO', tMs }, performance.now()));
  });
  trigger.start();
});
$('noise').addEventListener('input', (e) => {
  const value = Number((e.target as HTMLInputElement).value);
  trigger?.injectNoise(value); beam('noise', { sigma: value });
});

// --- FSM-Output → Hand + UI (der ≤125-ms-Pfad: synchron, kein await) ---
function apply(out: ReturnType<GraspFsm['dispatch']>): void {
  if (out.command === 'preshape' && fsm.hypothesis) hand.preshape(fsm.hypothesis);
  if (out.command === 'close' && fsm.hypothesis) {
    hand.close(fsm.hypothesis.force);
    if (reflexT0 !== null) {                                      // ≤125-ms-Beweis: direkt nach hand.close, live
      tiles.reflexMs = performance.now() - reflexT0; reflexT0 = null; renderTiles(hudTilesEl, tiles);
    }
    log.mark('grasp', performance.now());
  }
  if (out.command === 'open') hand.open();
  if (out.note === 'refusal-low-confidence' || out.note === 'no-grasp') log.mark('refusal', performance.now());
  const fsmBadge = $('fsm-state');
  fsmBadge.textContent = out.note ? `${out.state} · ${out.note}` : out.state;
  fsmBadge.dataset.tone = out.state === 'GRASP' ? 'ok'            // Zustand zusätzlich tonal (Text bleibt der Träger)
    : out.state === 'ARMED' || out.state === 'PRESHAPE' ? 'accent' : 'muted';
  renderStates(hudStatesEl, out.state);
  beam('fsm', { state: out.state, command: out.command, note: out.note });
}

/** Gemeinsamer Abschluss JEDES Hypothesen-Pfads (lokaler Snapshot + Studio-Frame): Hero-Zeile
 *  samt Ton, Meter/Tiles, Telemetrie, Episoden-Neustart, FSM-Dispatch, Timeout-Marker. Vorher
 *  copy-pasted in runSnapshot und applyStudioDetections — jede Rendering-Änderung musste zweimal
 *  gemacht werden. errorText ersetzt die Ergebnis-Zeile (Detektor-Fehlerpfad). */
function landHypothesis(h: GraspHypothesis, errorText?: string): void {
  hypothesisEl.textContent = errorText
    ?? `${h.objectLabel ?? '?'} → ${h.grip}/${h.force} @ ${h.confidence.toFixed(2)}${h.via === 'cnn' ? ' · CNN' : ''}`;
  hypothesisEl.dataset.tone = errorText || h.objectLabel === 'error' ? 'warn' : 'accent';   // Fehler nie im Hero-Gold
  hypothesisEl.title = '';                                       // frisches Ergebnis → alten Cortex/Edge-Tooltip löschen
  onHypothesis(h);                                               // Meter + Griff-Tile (+ zuvor gesetzte detectorMs-Tile)
  beam('hypothesis', { label: h.objectLabel, grip: h.grip, force: h.force, conf: Number(h.confidence.toFixed(3)), via: h.via });
  beginEpisode();                                                // GRASP? → erst öffnen; Replay neu → GO landet nach dem Reach
  apply(fsm.dispatch({ type: 'HYPOTHESIS', h }, performance.now()));
  lastHypothesisAt = performance.now();                          // Binding #2: ARMED/PRESHAPE-Timeout ab jetzt takten
}

/** Neue Foto-Episode (Handy-Frame ODER lokaler Snapshot = expliziter Nutzer-Akt für einen neuen Griff):
 *  hält die Hand gerade in GRASP, wird sie zuerst geöffnet (RELEASE_CMD) — die Drop-Safety gegen
 *  Vision-Rauschen bleibt intakt, nur der bewusste Foto-Akt darf lösen. Dann das EMG-Replay neu starten,
 *  damit GO erst nach dem Reach-Fenster landet statt sofort auf ein frisch armiertes ARMED. */
function beginEpisode(): void {
  if (fsm.state === 'GRASP') { beam('episode-release'); apply(fsm.dispatch({ type: 'RELEASE_CMD' }, performance.now())); }
  trigger?.stop(); trigger?.start();                             // start() setzt startedAt/lastIdx zurück; OnsetDetector-Instanz bleibt (Warmup ok)
}

// --- Snapshot-Fluss ---
navigator.mediaDevices?.getUserMedia({ video: { width: 640, height: 480 } })
  .then((s) => { video.srcObject = s; })
  .catch(() => hideWebcam());                                     // Kamera abgelehnt → Webcam-Pfad raus
if (!navigator.mediaDevices?.getUserMedia) hideWebcam();          // kein Kamera-API → nur Datei-Picker anbieten
function hideWebcam(): void {
  video.style.display = 'none';
  btnSnapshot.hidden = true;                                      // Snapshot-Button raus → Datei-Picker ist die eine klare Aktion
}

function refreshInputs(): void {
  const disabled = busy || manualOn;                             // Snapshot-Eingaben sperren während Analyse ODER im manuellen Modus
  btnSnapshot.disabled = disabled;
  filePick.disabled = disabled;
}

async function runSnapshot(source: string, draw: (ctx: CanvasRenderingContext2D) => void): Promise<void> {
  if (busy || manualOn) return;                                   // kein Doppelklick, kein Lauf im manuellen Modus
  busy = true; refreshInputs();
  beam('snapshot-start', { source });                            // Sichtbarkeit: markiert den Flussbeginn VOR jedem await
  try {
    const canvas = $('snap') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    draw(ctx);
    log.mark('snapshot', performance.now());
    log.mark('control_action', performance.now());                // der EINE Klick
    if (detectorStatus.state !== 'loading') hypothesisEl.textContent = 'analysiere…'; // Lade-% hat Vorrang
    lastCortexLabel = undefined;                                  // neuer Snapshot → alten Cortex-Vergleichschip zurücksetzen
    lastFrameTs = undefined;                                      // VOR dem await: invalidiert in-flight Studio-Inferenz sofort — deren
                                                                  // post-await-Guard darf das frische lokale Foto nicht mehr überstempeln
    const t0 = performance.now();
    const res = await analyzeSnapshot(canvas);
    tiles.detectorMs = performance.now() - t0;                    // Wall-Time um detect()//infer (nach Warmup ≈ reine Inferenz)
    const h = res.hypothesis;
    drawOverlay(ctx, res.ranked, res.chosenIdx, h.contactPoint);  // Tracking-Boxen NACH dem Bild aufs #snap-Canvas
    canvas.hidden = false;                                        // annotiertes Standbild sichtbar (im lokalen Demo sonst Offscreen-Buffer)
    applyGatesForDetector(activeGating, res.detector ?? 'rtdetr'); // Gates aufs tatsächliche Backend kalibrieren — VOR landHypothesis (Meter liest activeGating)
    landHypothesis(h, detectorStatus.state === 'failed' ? detectorErrorText(detectorStatus.msg) : undefined);
  } finally {
    busy = false; refreshInputs();
  }
}
btnSnapshot.addEventListener('click', () => { void runSnapshot('webcam', (ctx) => ctx.drawImage(video, 0, 0, 640, 480)); });
filePick.addEventListener('change', async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0]; if (!f) return;
  let bmp: ImageBitmap;
  try {
    bmp = await createImageBitmap(f);                             // dekodiert HEIC/große Handy-Fotos robuster als new Image()
  } catch {
    hypothesisEl.textContent = 'Bildformat nicht lesbar — bitte JPEG';
    beam('img-load-error', { type: f.type });
    return;
  }
  await runSnapshot('file', (ctx) => { ctx.drawImage(bmp, 0, 0, 640, 480); bmp.close(); });
});
$('btn-release').addEventListener('click', () => { log.mark('release', performance.now()); beam('release-click'); apply(fsm.dispatch({ type: 'RELEASE_CMD' }, performance.now())); });
$('eval-dl').addEventListener('click', (e) => { (e.target as HTMLAnchorElement).href = `data:application/json,${encodeURIComponent(log.toJson())}`; });

// --- Cortex (VLM) — der langsame, schlaue Zweitblick; überschreibt den Edge-Preshape per Soft-Lock ---
// Nur im Studio-Modus sichtbar. Kill-Switch ist rein lokal: AUS = Panel tot, kein Dispatch (Edge-Reflex läuft weiter).
interface CortexResult {
  object_label: string;
  grip: Grip;
  force: Force;
  contact_region: string;
  contact_point: { x: number; y: number };
  hazards: string[];
  rationale: string;
}
interface CortexMessage { type: 'cortex'; ts?: number; ok: boolean; result?: CortexResult; error?: string; ms?: number }

const cortexPane = $('cortex-pane');
const cortexBody = $('cortex-body');
const btnCortex = $('btn-cortex') as HTMLButtonElement;
let cortexOn = true;                                            // Kill-Switch (lokal, kein Server-Call)
let lastEdge: { label: string; grip: string } | null = null;   // Startpunkt der Vergleichszeile Edge → Cortex
let lastCortexLabel: string | undefined;                       // Cortex-Objektlabel des laufenden Frames → Overlay-Vergleichschip; pro Frame/Snapshot gelöscht
let lastFrameTs: number | undefined;                           // ts des zuletzt akzeptierten Studio-Frames: dedupliziert das Replay-on-Subscribe (SSE-Reconnect darf keine neue Episode starten) und verwirft Cortex-Ergebnisse älterer Frames
let redrawStudioOverlay: (() => void) | null = null;           // zeichnet das letzte Studio-Frame neu (Basisbild ohne Boxen → Overlay mit lastCortexLabel), sobald das Cortex-Event landet

function cortexChip(label: string, value: string): HTMLElement {
  const chip = document.createElement('span'); chip.className = 'cortex-chip';
  const l = document.createElement('span'); l.className = 'cortex-chip-label'; l.textContent = label;
  const v = document.createElement('span'); v.textContent = value;
  chip.append(l, v); return chip;
}
/** Panel auf einen einzelnen Text setzen (pending / idle / Fehler). */
function cortexMessage(text: string, cls = 'cortex-pending'): void {
  const p = document.createElement('div'); p.className = cls; p.textContent = text;
  cortexBody.replaceChildren(p);
}
function renderCortexPending(): void { cortexPane.classList.remove('is-dead'); cortexMessage('Cortex denkt…'); }
function renderCortexIdle(): void { cortexPane.classList.remove('is-dead'); cortexMessage('Cortex bereit — wartet auf Snapshot'); }
function renderCortexError(msg: string): void { cortexPane.classList.remove('is-dead'); cortexMessage(`Cortex-Fehler: ${msg}`, 'cortex-rationale'); }
/** Kill-Switch AUS: Panel grau/tot, Edge-Reflex läuft weiter. */
function renderCortexDead(): void {
  cortexPane.classList.add('is-dead');
  const badge = document.createElement('span'); badge.className = 'badge studio-dot'; badge.dataset.state = 'off'; badge.textContent = 'Cortex: getrennt';
  const msg = document.createElement('div'); msg.className = 'cortex-dead-msg'; msg.textContent = '35-€-Edge-Reflex läuft weiter';
  cortexBody.replaceChildren(badge, msg);
}
/** Volles Ergebnis: Griff/Kraft/Kontakt-Chips, Hazards + Advisory-Fineprint, Objekt+Rationale, Edge→Cortex-Vergleich. */
function renderCortexResult(r: CortexResult): void {
  cortexPane.classList.remove('is-dead');
  const frag = document.createDocumentFragment();
  const chips = document.createElement('div'); chips.className = 'cortex-row';
  chips.append(cortexChip('Griff', r.grip), cortexChip('Kraft', r.force), cortexChip('Kontakt', r.contact_region));
  frag.append(chips);
  if (r.hazards.length) {
    const hz = document.createElement('div'); hz.className = 'cortex-row';
    for (const h of r.hazards) { const c = document.createElement('span'); c.className = 'cortex-hazard'; c.textContent = `⚠ ${h}`; hz.append(c); }
    frag.append(hz);
    const fine = document.createElement('div'); fine.className = 'cortex-fineprint'; fine.textContent = '⚠ advisory insight — kein Sicherheitsfeature';
    frag.append(fine);
  }
  const obj = document.createElement('div'); obj.className = 'cortex-object'; obj.textContent = r.object_label; frag.append(obj);
  if (r.rationale) { const rat = document.createElement('div'); rat.className = 'cortex-rationale'; rat.textContent = r.rationale; frag.append(rat); }
  const cmp = document.createElement('div'); cmp.className = 'cortex-compare';
  const arrow = document.createElement('span'); arrow.className = 'cortex-arrow'; arrow.textContent = '→';
  cmp.append(document.createTextNode(`Edge: ${lastEdge ? `${lastEdge.label}/${lastEdge.grip}` : '—'}`), arrow, document.createTextNode(`Cortex: ${r.object_label}/${r.grip}`));
  frag.append(cmp);
  cortexBody.replaceChildren(frag);
}
/** Cortex-SSE-Event: Panel füllen und — wenn Kill-Switch AN und ok — als Hypothese durch apply() dispatchen. */
function handleCortex(msg: CortexMessage): void {
  if (msg.ts != null && msg.ts !== lastFrameTs) {              // Cortex eines ÄLTEREN Frames (neue Episode läuft schon) → verwerfen, sonst re-preshaped der alte Griff die neue Episode
    beam('cortex-stale', { ts: msg.ts });
    if (cortexBody.querySelector('.cortex-pending')) renderCortexIdle();   // hängendes „denkt…" einer toten Episode auflösen statt ewig warten
    return;
  }
  if (!cortexOn) { renderCortexDead(); return; }               // Kill-Switch AUS: Panel tot, kein Dispatch
  if (!msg.ok || !msg.result) { renderCortexError(msg.error ?? 'unbekannt'); return; }
  const r = msg.result;
  renderCortexResult(r);
  beam('cortex', { label: r.object_label, grip: r.grip, force: r.force, hazards: r.hazards });
  lastCortexLabel = r.object_label;                            // Overlay-Vergleichschip: Cortex-Label am gewählten Objekt zeigen
  if (!busy) redrawStudioOverlay?.();                          // Redraw stampft #snap-Pixel → nie während einer in-flight lokalen Analyse (Legacy-Frames ohne ts umgehen den Stale-Guard)
  const h: GraspHypothesis = {
    grip: r.grip, force: r.force, confidence: 0.9, source: 'cortex',
    objectLabel: r.object_label, contactRegion: r.contact_region, hazards: r.hazards, rationale: r.rationale,
  };
  onHypothesis(h);                                             // Meter + Griff-Tile auf den Cortex-Griff → Override sichtbar
  apply(fsm.dispatch({ type: 'HYPOTHESIS', h }, performance.now()));   // Soft-Lock überschreibt den Edge-Preshape
  lastHypothesisAt = performance.now();                        // wie jeder Hypothesen-Pfad: ARMED-Timeout ab jetzt takten
  if (fsm.state === 'ARMED' || fsm.state === 'PRESHAPE') {     // Dispatch akzeptiert → Cortex-Override prominent in die Badge
    hypothesisEl.textContent = `Cortex: ${r.object_label} — ${r.grip}/${r.force}`;
    hypothesisEl.title = lastEdge ? `Edge: ${lastEdge.label}/${lastEdge.grip}` : ''; // Edge-Herkunft als Tooltip erhalten
  } else if (fsm.state === 'GRASP') {                          // greift schon → HYPOTHESIS geschluckt (Drop-Safety); sagen, wie der neue Griff kommt
    const hint = document.createElement('div'); hint.className = 'cortex-fineprint';
    hint.textContent = '(hält — „Loslassen" übernimmt neuen Griff)';
    cortexBody.append(hint);
  }
}
/** Kill-Switch: rein lokal. AUS graut das Panel und stoppt den Dispatch; der Edge-Reflex läuft weiter. */
function setCortexKill(on: boolean): void {
  cortexOn = on;
  btnCortex.textContent = `Cortex: ${on ? 'AN' : 'AUS'}`;
  if (on) renderCortexIdle(); else renderCortexDead();
  beam('kill-switch', { on });
}

// --- Studio-Remote-Modus: SSE-Viewer (autonomer Pfad — Handy → Server /snap → /events → hier) ---
// Nur aktiv mit ?remote=. Frames tragen das Bild + ABSOLUTE Boxen der Handy-Auflösung; wir
// letterboxen ins 640×480-Canvas und transformieren die Boxen identisch → Downstream (rankDetections)
// sieht sie im Canvas-Raum. Kein analyzeSnapshot: die Detektionen kommen fertig vom Server.
interface StudioFrame {
  type?: 'frame';                                                 // fehlt bei Alt-Frames → als 'frame' behandeln (backward-compat)
  image_jpeg_b64: string;
  detections: Detection[];
  size?: [number, number];
  ms?: number;
  ts?: number;
  detector?: DetectorKind;                                        // Server-Backend, das die Boxen erzeugt hat (fehlt bei Alt-Frames → als 'rtdetr' behandelt)
}
const remoteEndpoint = getRemoteEndpoint();
if (remoteEndpoint) {
  ($('snap') as HTMLCanvasElement).hidden = false;                 // Frames sichtbar (im lokalen Demo bleibt der Canvas Offscreen-Buffer)
  video.style.display = 'none';                                    // Studio: das Handy liefert die Frames → lokale Webcam raus
  cortexPane.hidden = false; btnCortex.hidden = false;             // Cortex-Panel + Kill-Switch nur im Studio-Modus
  renderCortexIdle();
  btnCortex.addEventListener('click', () => setCortexKill(!cortexOn));
  const h2El = $('snapshot-pane').querySelector('h2');
  if (h2El) h2El.textContent = 'Sehen (Studio: Handy-Kamera)';
  const studioDot = document.createElement('span');
  studioDot.className = 'badge studio-dot';
  studioDot.dataset.state = 'off';
  studioDot.textContent = 'Studio: getrennt';
  h2El?.appendChild(studioDot);
  const setStudio = (connected: boolean): void => {
    studioDot.dataset.state = connected ? 'on' : 'off';
    studioDot.textContent = connected ? 'Studio: verbunden' : 'Studio: getrennt';
    beam('studio', { state: connected ? 'connected' : 'disconnected' });
  };
  const es = new EventSource(`${remoteEndpoint}/events`);
  es.addEventListener('open', () => setStudio(true));
  es.addEventListener('error', () => setStudio(false));
  es.addEventListener('message', (ev) => {
    if (manualOn) return;                                          // manuelle Baseline nicht durch Handy-Frames/Cortex verunreinigen
    let msg: StudioFrame | CortexMessage;
    try { msg = JSON.parse(ev.data) as StudioFrame | CortexMessage; }
    catch { beam('studio-frame-error', { msg: 'bad json' }); return; }
    if ((msg as CortexMessage).type === 'cortex') { handleCortex(msg as CortexMessage); return; }   // Cortex stampft keine #snap-Pixel — Redraw ist in handleCortex busy-gated, Staleness via ts-Guard
    if (busy) {                                                    // lokaler Snapshot in-flight → kein Frame darf die #snap-Pixel stampfen (der CNN liest sie noch)
      beam('studio-frame-busy-drop', { ts: (msg as { ts?: number }).ts });
      return;
    }
    const frame = msg as StudioFrame;                              // 'frame' oder (backward-compat) ohne type
    if (frame.ts != null && frame.ts === lastFrameTs) {            // Replay-on-Subscribe nach SSE-Reconnect: schon gesehen → KEINE neue Episode (sonst öffnet ein WLAN-Blip die haltende Hand)
      beam('studio-frame-dup', { ts: frame.ts });
      return;
    }
    lastFrameTs = frame.ts;
    drawStudioFrame(frame);
  });
}

/** Frame ins #snap-Canvas letterboxen und die Boxen mit demselben scale+offset mitziehen. */
function drawStudioFrame(frame: StudioFrame): void {
  const canvas = $('snap') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  redrawStudioOverlay = null;                                      // neues Frame beginnt zu dekodieren → alte Closure sofort tot, damit ein Cortex-Event nie ein vorheriges Frame neu malt
  const img = new Image();
  img.onload = () => {
    if (frame.ts != null && frame.ts !== lastFrameTs) {            // Decode-Reihenfolge invertiert (neueres Frame schon angewandt) → kein altes Bild drüberblitten
      beam('studio-frame-stale-decode', { ts: frame.ts });
      return;
    }
    if (manualOn) return;                                          // konnte während des Decodes umgeschaltet worden sein
    // frame.size (Original-Foto) MUSS vor naturalWidth (Preview ≤640) gewinnen — die Boxen
    // liegen im Original-Raum, das Preview ist nur das downgescalte Anzeige-JPEG.
    const { scale, drawW, drawH, offX, offY } = studioFrameGeometry(
      frame.size, img.naturalWidth, img.naturalHeight, canvas.width, canvas.height);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, offX, offY, drawW, drawH);
    const scaled: Detection[] = frame.detections.map((d) => ({
      label: d.label, score: d.score,
      box: {
        xmin: d.box.xmin * scale + offX, ymin: d.box.ymin * scale + offY,
        xmax: d.box.xmax * scale + offX, ymax: d.box.ymax * scale + offY,
      },
    }));
    const detector: DetectorKind = frame.detector ?? 'rtdetr';     // Alt-Frames ohne Feld → rtdetr (backward-compat)
    // applyStudioDetections ist jetzt async (der Student cropt+inferiert aus dem #snap-Canvas) → Promise
    // hier auflösen; die redrawStudioOverlay-Closure kapselt dasselbe ranked/chosenIdx wie zuvor.
    void applyStudioDetections(scaled, canvas.width, canvas.height, frame.ms, detector, frame.ts)
      .then((drawn) => {
        if (!drawn) return;                                        // stale Frame (neueres kam während der Inferenz) → keine Closure setzen, kein Repaint
        // Cortex kommt später als eigenes SSE-Event: dann Basisbild (ohne Boxen) neu blitten und Overlay mit lastCortexLabel zeichnen.
        redrawStudioOverlay = () => {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, offX, offY, drawW, drawH);
          drawOverlay(ctx, drawn.ranked, drawn.chosenIdx, drawn.contactPoint, lastCortexLabel);
        };
      })
      .catch((err: unknown) => beam('studio-frame-error', { msg: String((err as { message?: unknown })?.message ?? err) }));
  };
  img.onerror = () => beam('studio-frame-error', { msg: 'image decode' });
  img.src = `data:image/jpeg;base64,${frame.image_jpeg_b64}`;
}

/** Post-Detection-Fluss identisch zu runSnapshot, nur ohne analyzeSnapshot (Boxen sind fertig).
 *  Gibt die Render-Basis (ranked/chosenIdx/contactPoint) zurück, damit drawStudioFrame nach dem
 *  späteren Cortex-Event dasselbe Frame mit Edge→Cortex-Chip neu zeichnen kann. */
async function applyStudioDetections(dets: Detection[], w: number, h: number, ms: number | undefined, detector: DetectorKind, ts?: number): Promise<{ ranked: Detection[]; chosenIdx: number; contactPoint?: { x: number; y: number } } | null> {
  if (ts != null && ts !== lastFrameTs) {                          // schon bekannt stale (neueres Frame kam vor dem onload dran) → gar keine pre-await Seiteneffekte für ein totes Frame
    beam('studio-frame-stale-edge', { ts });
    return null;
  }
  applyGatesForDetector(activeGating, detector);                   // Gate-Kalibrierung auf den Detektor DIESES Frames — vor Meter-Render UND FSM-Dispatch (OVD-Scores liegen tiefer)
  lastCortexLabel = undefined;                                     // neuer Frame → Cortex-Vergleichschip zurücksetzen (bis das Cortex-Event dieses Frames landet)
  beam('snapshot-start', { source: 'phone' });
  log.mark('snapshot', performance.now());
  const snap = $('snap') as HTMLCanvasElement;                     // das Frame liegt schon drin → der Student cropt aus demselben Canvas
  const res = await analyzeDetailedWithStudent(dets, w, h, snap);
  if ((ts != null && ts !== lastFrameTs) || manualOn) {            // während der Student-Inferenz kam ein NEUERES Frame ODER der Nutzer schaltete manuell → Ergebnis ist stale.
    beam('studio-frame-stale-edge', { ts, manual: manualOn });     // VOR jedem post-await Seiteneffekt raus, sonst kontaminiert der alte Griff die neue Episode/Baseline
    return null;
  }
  const h2 = res.hypothesis;
  drawOverlay(snap.getContext('2d')!, res.ranked, res.chosenIdx, h2.contactPoint, lastCortexLabel); // Boxen NACH dem Frame-Bild; Cortex-Chip erst nach dem Cortex-Event
  if (ms != null) tiles.detectorMs = ms;                           // Detektor-Tile aus der Frame-ms des Servers (rendert in landHypothesis)
  landHypothesis(h2);
  lastEdge = { label: h2.objectLabel ?? '?', grip: h2.grip };      // Basis der Edge→Cortex-Vergleichszeile
  if (cortexOn) renderCortexPending();                             // "Cortex denkt…" bis das Cortex-Event landet
  return { ranked: res.ranked, chosenIdx: res.chosenIdx, contactPoint: h2.contactPoint };
}

// --- Manueller Modus (instrumentierte Rennen-Baseline: gleiche FSM/Hand, nur die Griff-Quelle wechselt) ---
const manual = new ManualMode();
function setMode(on: boolean): void {
  manualOn = on;
  $('btn-mode').textContent = `Modus: ${on ? 'manuell' : 'ayaka'}`;
  btnCycle.hidden = !on;                                          // Zyklus-Button nur im manuellen Modus
  refreshInputs();                                               // Snapshot-Fluss im manuellen Modus deaktiviert
  hypothesisEl.textContent = on ? 'Manueller Modus — Griff zyklen, dann GO' : '–';
  hypothesisEl.dataset.tone = 'muted';                           // Ton immer MIT dem Text setzen — sonst klebt das letzte accent/warn am Badge
}
$('btn-mode').addEventListener('click', () => setMode(!manualOn));
btnCycle.addEventListener('click', () => {
  manual.next();
  beam('manual-cycle', { grip: manual.current });
  log.mark('control_action', performance.now());                 // jeder Zyklus-Klick zählt — das ist der Vergleich zum EINEN Snapshot-Klick
  log.mark('mode_switch', performance.now());
  const mh: GraspHypothesis = { grip: manual.current, force: 'firm', confidence: 1, source: 'edge', objectLabel: 'manual' };
  onHypothesis(mh);
  apply(fsm.dispatch({ type: 'HYPOTHESIS', h: mh }, performance.now()));
  lastHypothesisAt = performance.now();                           // wie runSnapshot: ARMED-Timeout ab jetzt takten (identisches FSM-Verhalten)
});

// --- Loop ---
let last = performance.now();
(function loop(now: number) {
  hand.tick(Math.min(now - last, 50)); last = now;                // Binding #1: dt clampen — großer dt (Tab-Resume) ließe die Hand divergieren
  if (trigger) { trigger.tick(now); chart.push(trigger.mav); }    // Binding #3: chart plottet Rohsignal trigger.mav (Detektor glättet intern → Fineprint)
  if (lastHypothesisAt !== null && (fsm.state === 'ARMED' || fsm.state === 'PRESHAPE')
      && now - lastHypothesisAt > activeGating.armedTimeoutMs) {   // Binding #2: FSM ist timer-los → TIMEOUT hier takten (aktives armedTimeoutMs), sonst öffnet eine armierte Hand nie
    apply(fsm.dispatch({ type: 'TIMEOUT' }, now));
    lastHypothesisAt = null;
  }
  chart.draw();
  controls.update();                                              // OrbitControls-Damping (dampingFactor 0.08)
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
})(last);
