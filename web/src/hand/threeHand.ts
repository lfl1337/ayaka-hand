import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { Force, GraspHypothesis, Hand, HandPose } from '../types';
import { GRIP_PRESETS, closedCurls, type FingerCurls } from './presets';

interface HandCfg { preshapeMs: number; closeMs: number; releaseMs: number; delicateCurl: number }

const FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky'] as const;

/** Blender-gebautes Prothesen-Modell (web/scripts/blender/hand_build.py) — geriggt + geskinnt,
 *  Bones `{finger}_{glied}`, Materialien by name (shell/joint/lens). Selbes Weltmaß wie das alte Rig. */
const MODEL_URL = '/models/hand/ayaka-hand.glb';

/** Ladefehler sichtbar machen: Konsole + CustomEvent (main.ts beamt + zeigt Warn-Badge) —
 *  eine leere Bühne ohne Signal wäre der schlimmste Demo-Fehlermodus. */
function emitHandError(msg: string): void {
  console.error('hand-model', msg);
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('ayaka:hand-model-error', { detail: msg }));
  }
}

/** Handfarbe aus dem CSS-Token --hand-color (Light/Dark), einmal beim attach gelesen.
 *  THREE.Color parst den Hex-String direkt; Fallback fürs Headless-Rendering. */
function handColorFromCss(): THREE.Color {
  const fallback = '#8fb4c9';
  const raw = typeof document !== 'undefined'
    ? getComputedStyle(document.documentElement).getPropertyValue('--hand-color').trim()
    : '';
  return new THREE.Color(raw || fallback);
}

/** Technischer Dunkelton für Gelenk-/Sockel-Flächen aus --hand-joint (Light/Dark),
 *  analog zu handColorFromCss; Fallback fürs Headless-Rendering. */
function handJointColorFromCss(): THREE.Color {
  const fallback = '#3d4852';
  const raw = typeof document !== 'undefined'
    ? getComputedStyle(document.documentElement).getPropertyValue('--hand-joint').trim()
    : '';
  return new THREE.Color(raw || fallback);
}

/** Bewegungslogik ist reines tick(dt)-Lerping auf Ziel-Curls → headless testbar (joints bleiben leer).
 *  Rendering (attachTo) lädt das glb asynchron und mappt die Bones in joints[fi][si]; bis dahin
 *  no-opt applyToJoints über den Leer-Guard. */
export class ThreeHand implements Hand {
  private cur: FingerCurls = { ...GRIP_PRESETS.no_grasp };
  private target: FingerCurls = { ...GRIP_PRESETS.no_grasp };
  private durationMs = 1;
  private _state: HandPose = 'open';
  private grip: GraspHypothesis['grip'] = 'no_grasp';
  private joints: THREE.Object3D[][] = [];
  private rest: THREE.Quaternion[][] = [];

  // Bone-Lokalachsen des Blender-Exports: +Y läuft den Finger entlang, +X-Rotation krümmt zur
  // Greifseite (in den Pose-Renders des Build-Scripts verifiziert). Spread = Twist um die Fingerachse,
  // Vorzeichen invertiert zum alten −Z-Finger-Rig.
  private static readonly AXIS_FLEX = new THREE.Vector3(1, 0, 0);
  private static readonly AXIS_SPREAD = new THREE.Vector3(0, 1, 0);
  private readonly tmpQ = new THREE.Quaternion();
  private readonly tmpQ2 = new THREE.Quaternion();

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

  /** Lädt das geriggte glb, ersetzt die exportieren Materialien durch Token-getriebene (Light/Dark),
   *  mappt Bones by name in joints[fi][si] und merkt sich die Rest-Quaternions — applyToJoints
   *  komponiert die Pose auf die Rest-Pose (Bones haben, anders als die alten Pivots, eine
   *  Ausrichtung im Rig, die eine absolute Euler-Zuweisung zerstören würde). */
  attachTo(scene: THREE.Scene): void {
    const shellColor = handColorFromCss();
    const jointColor = handJointColorFromCss();
    const mats: Record<string, THREE.Material> = {
      shell: new THREE.MeshPhysicalMaterial({ color: shellColor, roughness: 0.48, metalness: 0.08, clearcoat: 0.65, clearcoatRoughness: 0.28 }),
      joint: new THREE.MeshStandardMaterial({ color: jointColor, roughness: 0.45, metalness: 0.3 }),
      lens: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#1d6a76'), roughness: 0.15, metalness: 0.6,
        emissive: new THREE.Color('#0d3a42'), emissiveIntensity: 0.7,
      }),
    };

    // Boden: kreisförmige Schattenebene (nur der Schatten ist sichtbar) + sehr dezentes Grid.
    const ground = new THREE.Mesh(new THREE.CircleGeometry(3.2, 48), new THREE.ShadowMaterial({ opacity: 0.28 }));
    ground.rotation.x = -Math.PI / 2; ground.position.y = -0.55; ground.receiveShadow = true;
    scene.add(ground);
    const grid = new THREE.GridHelper(10, 10);
    const gridMat = grid.material as THREE.LineBasicMaterial;
    gridMat.transparent = true; gridMat.opacity = 0.12;
    grid.position.y = -0.549;                                            // minimal über der Ebene → kein z-Fighting
    scene.add(grid);

    new GLTFLoader().load(
      MODEL_URL,
      (gltf) => {
        gltf.scene.traverse((o) => {
          const mesh = o as THREE.Mesh;
          if (!mesh.isMesh) return;
          mesh.castShadow = true; mesh.receiveShadow = true;
          mesh.frustumCulled = false;                                    // SkinnedMesh-Bounds folgen den Bones nicht
          const name = (mesh.material as THREE.Material)?.name ?? '';
          if (mats[name]) mesh.material = mats[name];
        });
        const chains: THREE.Object3D[][] = [];
        const rests: THREE.Quaternion[][] = [];
        for (const f of FINGERS) {
          const chain: THREE.Object3D[] = [];
          const rest: THREE.Quaternion[] = [];
          for (let s = 0; s < (f === 'thumb' ? 2 : 3); s++) {
            const b = gltf.scene.getObjectByName(`${f}_${s}`);
            if (!b) { emitHandError(`bone ${f}_${s} fehlt — Rig unbrauchbar`); return; }
            chain.push(b); rest.push(b.quaternion.clone());
          }
          chains.push(chain); rests.push(rest);
        }
        this.joints = chains; this.rest = rests;
        scene.add(gltf.scene);
      },
      undefined,
      (err) => emitHandError(String((err as { message?: unknown })?.message ?? err)),
    );
  }

  private applyToJoints(): void {
    if (!this.joints.length) return;
    FINGERS.forEach((name, fi) => {
      const curl = this.cur[name];
      const isThumb = name === 'thumb';
      this.joints[fi].forEach((bone, si) => {
        const flex = si === 1 ? 0.06 : si === 2 ? 0.03 : 0;             // konstanter Ruhe-Mikroflex am Mittel-/Endglied (rein optisch)
        const ang = (isThumb ? 0.9 : 1.15) * curl * (si === 0 ? 0.8 : 1.0) + flex;
        this.tmpQ.setFromAxisAngle(ThreeHand.AXIS_FLEX, ang);
        if (si === 0) {
          const spread = -(fi - 2) * 0.12 * this.cur.spread * (isThumb ? 3 : 1);
          this.tmpQ2.setFromAxisAngle(ThreeHand.AXIS_SPREAD, spread);
          this.tmpQ.multiply(this.tmpQ2);
        }
        bone.quaternion.copy(this.rest[fi][si]).multiply(this.tmpQ);
      });
    });
  }
}
