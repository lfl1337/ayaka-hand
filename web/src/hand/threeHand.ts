import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import type { Force, GraspHypothesis, Hand, HandPose } from '../types';
import { GRIP_PRESETS, closedCurls, type FingerCurls } from './presets';

interface HandCfg { preshapeMs: number; closeMs: number; releaseMs: number; delicateCurl: number }

const FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky'] as const;

/** Handfarbe aus dem CSS-Token --hand-color (Light/Dark), einmal beim attach gelesen.
 *  THREE.Color parst den Hex-String direkt; Fallback fürs Headless-Rendering. */
function handColorFromCss(): THREE.Color {
  const fallback = '#8fb4c9';
  const raw = typeof document !== 'undefined'
    ? getComputedStyle(document.documentElement).getPropertyValue('--hand-color').trim()
    : '';
  return new THREE.Color(raw || fallback);
}

/** Technischer Dunkelton für Gelenke/Knöchel + Handflächen-Pad aus --hand-joint (Light/Dark),
 *  analog zu handColorFromCss; Fallback fürs Headless-Rendering. */
function handJointColorFromCss(): THREE.Color {
  const fallback = '#3d4852';
  const raw = typeof document !== 'undefined'
    ? getComputedStyle(document.documentElement).getPropertyValue('--hand-joint').trim()
    : '';
  return new THREE.Color(raw || fallback);
}

const TAPER = 0.82;               // Radius-Verjüngung Glied→Glied (Spitze dünner als Basis)
const RSEG = 16;                  // radiale Segmente pro Zylinder (≤24 → sane Polycount)

/** Relative Fingergeometrie: Wurzelposition an der Handkante, Gliedlängen (proximal→distal),
 *  Basisradius. Mittelfinger am längsten, Pinky ~78 %, Radien variiert (Pinky am dünnsten). */
const FINGER_SPECS: Record<'index' | 'middle' | 'ring' | 'pinky', { rootX: number; rootZ: number; lens: number[]; r: number }> = {
  index:  { rootX: -0.21, rootZ: -0.42, lens: [0.34, 0.27, 0.20], r: 0.058 },
  middle: { rootX:  0.00, rootZ: -0.46, lens: [0.39, 0.30, 0.22], r: 0.060 },
  ring:   { rootX:  0.20, rootZ: -0.43, lens: [0.36, 0.28, 0.21], r: 0.054 },
  pinky:  { rootX:  0.38, rootZ: -0.35, lens: [0.30, 0.24, 0.17], r: 0.045 },
};

/** Baut eine Fingerkette: verjüngte Zylinder-Glieder (Spitze = TAPER×Basis), Knöchelkugel je Gelenk,
 *  gerundete Kuppe am letzten Glied. Rückgabe = Pivot-Kette (Index 0 = Wurzel), von applyToJoints gedreht. */
function buildFinger(
  lens: number[], rootRadius: number,
  shell: THREE.Material, joint: THREE.Material, pad: THREE.Material,
): THREE.Object3D[] {
  const pivots: THREE.Object3D[] = [];
  let baseR = rootRadius;
  lens.forEach((len, s) => {
    const tipR = baseR * TAPER;
    const pivot = new THREE.Object3D();
    if (s > 0) { pivot.position.z = -lens[s - 1]; pivots[s - 1].add(pivot); }   // Gelenk sitzt am Ende des Vorgänger-Glieds
    // Glied: dick am Gelenk (z=0), dünn zur Spitze (z=-len). rotation.x=π/2 dreht die Zylinderachse Y→Z.
    const seg = new THREE.Mesh(new THREE.CylinderGeometry(baseR, tipR, len, RSEG), shell);
    seg.rotation.x = Math.PI / 2; seg.position.z = -len / 2;
    seg.castShadow = true; seg.receiveShadow = true;
    const knuckle = new THREE.Mesh(new THREE.SphereGeometry(baseR * 1.05, 12, 8), joint);   // Knöchelkappe, überdeckt die Naht
    knuckle.castShadow = true; knuckle.receiveShadow = true;
    pivot.add(seg, knuckle);
    if (s === lens.length - 1) {                                                // gerundete Fingerkuppe (etwas dunkler als Shell)
      const tip = new THREE.Mesh(new THREE.SphereGeometry(tipR, 12, 8), pad);
      tip.position.z = -len; tip.castShadow = true; tip.receiveShadow = true;
      pivot.add(tip);
    }
    pivots.push(pivot);
    baseR = tipR;                                                              // stetige Verjüngung über die Glieder
  });
  return pivots;
}

/** Bewegungslogik ist reines tick(dt)-Lerping auf Ziel-Curls → headless testbar.
 *  Rendering (attachTo) baut ein Prothesen-Rig (verjüngte Glieder, Knöchel, Sockel) und liest nur `curls`. */
export class ThreeHand implements Hand {
  private cur: FingerCurls = { ...GRIP_PRESETS.no_grasp };
  private target: FingerCurls = { ...GRIP_PRESETS.no_grasp };
  private durationMs = 1;
  private _state: HandPose = 'open';
  private grip: GraspHypothesis['grip'] = 'no_grasp';
  private joints: THREE.Object3D[][] = [];

  private cfg: HandCfg;
  constructor(cfg: HandCfg) { this.cfg = cfg; }

  get state(): HandPose { return this._state; }
  get curls(): FingerCurls { return { ...this.cur }; }

  preshape(h: GraspHypothesis): void {
    this.grip = h.grip;
    this.target = { ...GRIP_PRESETS[h.grip] };
    this.durationMs = this.cfg.preshapeMs;
    this._state = 'preshaped';
  }
  close(force: Force): void {
    this.target = closedCurls(this.grip, force, this.cfg.delicateCurl);
    this.durationMs = this.cfg.closeMs;
    this._state = 'closed';
  }
  open(): void {
    this.grip = 'no_grasp';
    this.target = { ...GRIP_PRESETS.no_grasp };
    this.durationMs = this.cfg.releaseMs;
    this._state = 'open';
  }

  tick(dtMs: number): void {
    const k = Math.min(1, dtMs / this.durationMs);
    for (const f of [...FINGERS, 'spread'] as const) {
      this.cur[f] += (this.target[f] - this.cur[f]) * (3 * k);            // exponentielles Zulaufen
      if (Math.abs(this.target[f] - this.cur[f]) < 0.002) this.cur[f] = this.target[f];
    }
    this.applyToJoints();
  }

  /** Prothesen-Rig: gerundete Handfläche (RoundedBox) + Handgelenk-Sockel, 5 Finger aus verjüngten
   *  Gliedern mit Knöchelkappen und gerundeten Kuppen, Daumen an einem Opposition-Mount.
   *  Shell = --hand-color, Gelenke/Palm-Pad = --hand-joint; Meshes werfen/empfangen Schatten;
   *  darunter eine ShadowMaterial-Bodenscheibe + dezentes Grid als Bühne. */
  attachTo(scene: THREE.Scene): void {
    const shellColor = handColorFromCss();
    const jointColor = handJointColorFromCss();
    // Shell mit dezentem Clearcoat (Schalen-Look); Gelenke/Pad im dunklen Technik-Ton; Kuppen etwas dunkler als Shell.
    const shell = new THREE.MeshPhysicalMaterial({ color: shellColor, roughness: 0.55, metalness: 0.08, clearcoat: 0.6, clearcoatRoughness: 0.3 });
    const joint = new THREE.MeshStandardMaterial({ color: jointColor, roughness: 0.5, metalness: 0.25 });
    const pad = new THREE.MeshStandardMaterial({ color: shellColor.clone().multiplyScalar(0.8), roughness: 0.6, metalness: 0.05 });

    // Handfläche: gerundeter Block statt Kiste, dezent geflacht (Wedge auf die Geometrie gebacken → Kinder unverzerrt).
    const palmGeo = new RoundedBoxGeometry(0.86, 0.17, 0.92, 4, 0.06);
    palmGeo.scale(1, 0.9, 1);
    const palm = new THREE.Mesh(palmGeo, shell);
    palm.castShadow = true; palm.receiveShadow = true;
    scene.add(palm);

    // Inneres Handflächen-Pad (Gelenk-Ton) auf der Greifseite (+y, dorthin schließen die Finger).
    const padPlate = new THREE.Mesh(new RoundedBoxGeometry(0.62, 0.04, 0.66, 3, 0.018), joint);
    padPlate.position.set(0, 0.082, -0.03); padPlate.castShadow = true; padPlate.receiveShadow = true;
    palm.add(padPlate);

    // Prothesen-Handgelenk: kurzer konischer Sockel + Ringband am Übergang (Gelenk-Ton).
    const socket = new THREE.Mesh(new THREE.CylinderGeometry(0.30, 0.34, 0.18, 24), joint);
    socket.rotation.x = Math.PI / 2; socket.position.set(0, -0.01, 0.54);
    socket.castShadow = true; socket.receiveShadow = true;
    palm.add(socket);
    const collar = new THREE.Mesh(new THREE.TorusGeometry(0.32, 0.03, 10, 24), joint);
    collar.rotation.x = Math.PI / 2; collar.position.set(0, 0, 0.46);
    collar.castShadow = true; collar.receiveShadow = true;
    palm.add(collar);

    // Boden: kreisförmige Schattenebene (nur der Schatten ist sichtbar) + sehr dezentes Grid.
    const ground = new THREE.Mesh(new THREE.CircleGeometry(3.2, 48), new THREE.ShadowMaterial({ opacity: 0.28 }));
    ground.rotation.x = -Math.PI / 2; ground.position.y = -0.55; ground.receiveShadow = true;
    scene.add(ground);
    const grid = new THREE.GridHelper(10, 10);
    const gridMat = grid.material as THREE.LineBasicMaterial;
    gridMat.transparent = true; gridMat.opacity = 0.15;
    grid.position.y = -0.549;                                            // minimal über der Ebene → kein z-Fighting
    scene.add(grid);

    // Finger in FINGERS-Reihenfolge (joints-Index = fi, wie applyToJoints erwartet).
    FINGERS.forEach((name) => {
      if (name === 'thumb') {
        // Daumen: tiefer/vorn an der Palm-Seite, Basisrotation ~40° → natürliche Opposition zur Handfläche.
        const mount = new THREE.Object3D();
        mount.position.set(-0.40, -0.01, 0.16);
        mount.rotation.set(0.2, -0.8, 0.42);
        palm.add(mount);
        const pivots = buildFinger([0.34, 0.28], 0.075, shell, joint, pad);   // 2 Glieder, dickster Radius
        mount.add(pivots[0]);
        this.joints.push(pivots);
      } else {
        const sp = FINGER_SPECS[name];
        const pivots = buildFinger(sp.lens, sp.r, shell, joint, pad);
        pivots[0].position.set(sp.rootX, 0, sp.rootZ);
        palm.add(pivots[0]);
        this.joints.push(pivots);
      }
    });
  }

  private applyToJoints(): void {
    if (!this.joints.length) return;
    FINGERS.forEach((name, fi) => {
      const curl = this.cur[name];
      const isThumb = name === 'thumb';
      this.joints[fi].forEach((pivot, si) => {
        const flex = si === 1 ? 0.06 : si === 2 ? 0.03 : 0;             // konstanter Ruhe-Mikroflex am Mittel-/Endglied (rein optisch)
        pivot.rotation.x = (isThumb ? 0.9 : 1.15) * curl * (si === 0 ? 0.8 : 1.0) + flex;
        if (si === 0) pivot.rotation.z = (fi - 2) * 0.12 * this.cur.spread * (isThumb ? 3 : 1);
      });
    });
  }
}
