# ayaka-hand

**An AI-assisted hand/arm prosthesis that pre-shapes the grip *before* contact — with a working, public in-browser demo.**

A wrist camera looks at the object you are reaching for and forms the correct grasp on the way in — *vision-preshaping*. Your own muscle signal (EMG) is reduced to a single binary **GO** trigger: you decide *when* to close, the vision system decides *how*. The goal is to remove the moment-to-moment cognitive load that drives real-world prosthesis abandonment.

> **Status:** research prototype / hackathon submission (AMD Developer Hackathon, Act II). Not a medical device, and **no physical hand has been built yet** — the demo renders a simulated hand in the browser (three.js). See [Safety & hazard disclaimer](#safety--hazard-disclaimer).

---

## Why

Roughly **23 %** of upper-limb myoelectric prostheses are abandoned in real-world use. A recurring reason is *cognitive load*: conventional myoelectric control asks the user to consciously modulate muscle contractions to select a grip and meter force, for every single object, all day. It is exhausting.

ayaka-hand moves the "how to grip" decision off the user and onto a camera + a small neural network:

- **The user** provides intent: a binary GO (close) / release, from one EMG channel.
- **The hand** provides dexterity: it has already *seen* the cup / knife / key and pre-shaped the fingers into a power / lateral / pinch / tripod grasp with an appropriate force level before contact.

## How it works — a two-tier "teacher / reflex" architecture

The core design constraint is latency. A grasp must be committed within a **≤125 ms** control window to feel like part of your body. A cloud vision-language model cannot live inside that loop — so it doesn't.

```
                         ┌─────────────────────────────────────────────┐
   OFFLINE / ADVISORY    │  CORTEX  —  cloud VLM (Qwen3-VL-32B-Instruct) │
   (never in the loop)   │  • Teacher: labels training data              │
                         │  • Live demo: "second opinion" panel, ~2–3 s  │
                         │  • Display-only. NEVER drives the actuator.   │
                         └───────────────────────┬─────────────────────┘
                                 distillation     │  (offline)
                                 grip + force ▼    │
   REAL-TIME  ┌──────────────────────────────────────────────────────┐
   ≤125 ms    │  STUDENT  —  on-device CNN reflex (2.23M params)       │
   loop       │  • MobileNetV2 dual-head: grip (5) + force (2)          │
              │  • Runs on wrist camera pixels only                    │
              │  • THE real-time decision-maker                        │
              └───────────────┬──────────────────────────────────────┘
                              │ pre-shape
   EMG (1 ch) ── GO trigger ──┴──▶  actuated hand closes on commit
```

**Cortex** (the teacher). A large vision-language model — [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen) (Apache-2.0) — reasons about object appearance and state and emits a structured grasp label (`grip`, `force`, `contact_region`, `contact_point`, `hazards`, `rationale`). It is used **offline** to label the training set, and in the live demo as a slower "second opinion" panel that appears a couple of seconds after the grasp. **It is never in the real-time control loop.**

**Student** (the reflex). A **2,232,839-parameter** (≈ **2.23M**) MobileNetV2 with two heads (grip + force), distilled from the Cortex labels. This is the network that actually runs on the wrist camera in real time. It sees only pixels — no ground-truth class name, no object detector telling it "this is a cup." In the browser demo it runs as an 8.9 MB ONNX model via ONNX-Runtime Web on the **WASM** backend (WebGPU is deliberately disabled — it fails on several mobile browsers).

**Schema v1 (frozen).** Grips: `[power, lateral, pinch, tripod, no_grasp]`. Forces: `[delicate, firm]`. The index order is load-bearing and identical across the Python schema, the TypeScript constants, and the ONNX metadata.

## The measured result

The headline claim is that a tiny pixels-only student can **beat a much stronger-looking lookup baseline** — and it does, on every metric. All numbers below are the project's single source of truth: [`docs/benchmarks/distill-before-after.md`](docs/benchmarks/distill-before-after.md), measured on **613 validation teacher-labels**.

The baseline is deliberately generous: it is handed the **ground-truth object class name** (the strongest possible strawman — a perfect detector) and looks up a canonical grip. The student gets none of that; it only sees the crop pixels.

| Metric | Lookup baseline *(gets GT class name)* | Student *(2.23M, pixels only)* |
|---|---:|---:|
| Grip accuracy | 70.5 % | **72.3 %** |
| Grip macro-F1 | 0.581 | **0.659** |
| Force accuracy | 65.4 % | **81.1 %** |
| Force macro-F1 | 0.595 | **0.806** |

Per-class grip F1 — the student wins hardest exactly where a class-name lookup is *structurally blind*:

| Grip class | Baseline F1 | Student F1 |
|---|---:|---:|
| power | 0.722 | 0.752 |
| lateral | 0.766 | 0.702 |
| pinch | **0.000** | **0.323** |
| tripod | 0.594 | 0.727 |
| no_grasp | 0.823 | 0.790 |

The lookup baseline scores **0.000** on `pinch` because no single object class maps cleanly to a pinch — you have to *look*. The student also gains **+16 points of force accuracy**: force depends on the object's *state* (a full cup vs. an empty one), which a class name cannot encode. "Sees form **and** state" is a measurement here, not a slogan.

**Honest ablation** (same val set, same pipeline — full table in the benchmark doc):

| Run | Architecture | Params | Init | Grip Acc | Grip F1 | Force Acc | Force F1 |
|---|---|---:|---|---:|---:|---:|---:|
| kd-v1 | MobileNetV2-0.75 | 1.36M | scratch | 49.9 % | 0.484 | 74.4 % | 0.741 |
| kd-v2 | MobileNetV2-0.75 | 1.36M | scratch (200 ep) | — | 0.497* | — | — |
| **kd-v3 (chosen)** | **MobileNetV2-1.0 (timm)** | **2.23M** | **ImageNet** | **72.3 %** | **0.659** | **81.1 %** | **0.806** |
| kd-v4 | MobileNetV2-0.5 (timm) | 0.70M | ImageNet | 69.5 % | 0.635 | 81.6 % | 0.812 |

\* kd-v2: best val grip macro-F1 from the training log; no full eval since it was below baseline.

From-scratch loses to the lookup strawman on grip (11k images is too few); ImageNet initialization flips it. The parameter count was corrected *downward* over the project's history (an earlier "2.6M" wrongly counted the discarded ImageNet classifier head; the from-scratch 0.75-width model was 1.36M) — the communicated figure is **2.23M**, and we do not reuse the old numbers. Even the 0.70M kd-v4 beats the baseline on grip macro-F1.

Training: RTX 5090, bf16, weighted cross-entropy (`grip + 0.5·force`), AdamW, cosine schedule, 11,004 teacher labels, Prompt v1.1. ONNX export (kd-v3): 8.9 MB, ONNX-Runtime parity Δ ≈ 6e-6.

## Repository layout

```
web/                    Browser demo (TypeScript + Vite). The judge-facing app.
  src/                    FSM, perception, hand rendering, EMG trigger, UI
  public/models/          Bundled ONNX: student + D-FINE + RT-DETRv2 detectors
  public/ort/             Self-hosted ONNX-Runtime WASM (works offline, no CDN)
  scripts/                fetch-detector.sh, smoke tests, telemetry sink
factory/                Python "factory" — how the models are made
  distill/                Teacher→student distillation pipeline (+ 21 tests)
  serve/                  FastAPI inference server for the live "second opinion"
  labeling/               COCO harvest + VLM batch labeling
  probe/                  Cortex prompt + probe scripts
  schema/                 grasp_schema_v1.json (the frozen contract)
data/ayaka-crops/       Teacher labels + COCO manifests (attribution). NO raw images.
docs/                   Benchmarks (source of truth) + dataset provenance
```

## Setup & run

### 1. Browser demo (recommended — runs fully client-side)

The demo is intentionally **pure-local**: with no inference server it runs entirely in the browser (D-FINE detector + Student CNN, with a lookup fallback if the student fails to load). Nothing to 500.

```bash
cd web
pnpm install
pnpm build
pnpm exec vite preview --host 0.0.0.0 --port 4173
# open http://localhost:4173/
```

The student and detector ONNX models and the ONNX-Runtime WASM are **bundled** under `web/public/` so a fresh clone runs offline. To re-fetch the detectors from Hugging Face instead, run `bash scripts/fetch-detector.sh` from `web/`. (Requires Node + [pnpm](https://pnpm.io/).)

### 2. Inference server — optional, enables the Cortex "second opinion"

Only needed if you want the live VLM second-opinion panel (`?remote=1`). Requires an [OpenRouter](https://openrouter.ai/) API key.

```bash
cd factory/serve
cp ../../.env.example ../../.env      # then put your key in .env (repo root)
uv sync
uv run uvicorn server:app --host 127.0.0.1 --port 27461
curl -s localhost:27461/health
```

The open-vocabulary detector (LLMDet) is downloaded from Hugging Face on first startup; it falls back to a bundled RT-DETRv2 path if that load fails.

> **⚠ Security — this server is NOT production-ready.** It currently sets CORS `allow_origins=["*"]` and has **no authentication on any endpoint** (`/health`, `/infer`, `/snap`, `/cortex-toggle`, `/events`). It is safe only for `127.0.0.1` / trusted-tunnel use. **Do not expose it to a public network.** See [Known limitations](#known-limitations).

### 3. Reproduce the student (distillation pipeline)

Requires [uv](https://docs.astral.sh/uv/), a CUDA GPU, and the COCO crops (see [Data & attribution](#data--attribution)).

```bash
cd factory/distill
uv sync
uv run pytest                                    # 21 tests

# build index → train (winner arch) → evaluate → export ONNX
uv run python -m distill.build_dataset --data-root <crops> --labels <labels.jsonl> --out index.json
uv run python -m distill.train    --data-root <crops> --train-labels <train.jsonl> --val-labels <val.jsonl> \
                                  --out-dir runs/kd-v3 --arch timm_mnv2_100 --epochs 80
uv run python -m distill.evaluate --data-root <crops> --val-labels <val.jsonl> \
                                  --checkpoint runs/kd-v3/best.pt --out-md table.md --out-json results.json
uv run python -m distill.export_onnx --checkpoint runs/kd-v3/best.pt --out runs/kd-v3/student.onnx
```

## Models & provenance

| Model | Role | Params / size | Upstream & license |
|---|---|---|---|
| **Student** (`web/public/models/student/`) | real-time grip+force reflex | 2.23M / 8.9 MB ONNX | Our own weights. Backbone: timm `mobilenetv2_100.ra_in1k` (**Apache-2.0**) |
| **D-FINE-S** (`.../dfine/`) | in-browser object detector (primary) | quantized ONNX | `onnx-community/dfine_s_coco-ONNX` (**Apache-2.0**) |
| **RT-DETRv2-r18** (`.../rtdetr/`) | in-browser detector (fallback) | quantized ONNX | `onnx-community/rtdetr_v2_r18vd-ONNX` (**Apache-2.0**) |
| **ONNX-Runtime Web** (`web/public/ort/`) | WASM inference runtime | 4 files, ~36 MB | Microsoft `onnxruntime-web` (**MIT**) |
| **Cortex / teacher** (not bundled) | offline labeling + demo second opinion | 32B, cloud | Qwen3-VL-32B-Instruct (**Apache-2.0**), via OpenRouter |

The student ONNX is bundled for convenience but is fully reproducible from `factory/distill`. Its provenance/signature (opset 17, input `[batch,3,160,160]`, outputs `grip_logits[.,5]` + `force_logits[.,2]`) is documented in `web/public/models/student/README.md`.

## Data & attribution

The teacher-labeled training set is derived from **COCO 2017** object crops (ground-truth bounding boxes over ~22 graspable household classes + 4 "no-grasp" anchor classes). Full rationale and parameters are in [`docs/data/DATASET.md`](docs/data/DATASET.md).

**License filtering is applied per image.** Only COCO license IDs `{4, 5, 7, 8}` are kept; all **NonCommercial** (1–3) and **NoDerivatives** (6) images are **excluded**, because the crops are derivative works. The crops actually used break down as:

| COCO license | License | Crops |
|---|---|---:|
| 4 | CC BY 2.0 (Attribution) | 7,507 |
| 5 | CC BY-SA 2.0 (Attribution-ShareAlike) | 4,103 |
| 7 | No known copyright restrictions | 7 |
| | **Total** | **11,617** (11,004 train / 613 val) |

_(License ID 8 — US Government Work — is also whitelisted but matched no crops in these classes, so it is not listed above.)_

> **Note on redistribution.** This repository **does not ship the COCO-derived crop images.** ~4,103 of them are CC **BY-SA** (a copyleft/ShareAlike term), and redistributing the pixels would entangle the whole dataset in ShareAlike obligations. Instead we ship, in `data/ayaka-crops/`:
> - **`manifest-{train,val}2017.jsonl`** — per-crop attribution: source `flickr_url`, `coco_url`, `license`, COCO image/annotation IDs, and the bounding box. This is the full recipe to *reproduce* the exact crops.
> - **`labels-{train,val}2017-backup.jsonl`** — the teacher-generated grip/force annotations (our own contribution).
>
> To regenerate the images from the manifests, use `factory/labeling/harvest_coco.py` (re-downloads from `images.cocodataset.org` and re-applies the license filter). Per-image attribution is carried in the manifests as required by CC BY / BY-SA.

## Safety & hazard disclaimer

**This is a research prototype, not a certified safety device, and it makes no injury-prevention claims.**

The Cortex VLM can emit advisory `hazards` flags (e.g. `hot`, `sharp`, `slippery`, `fragile`, `full`, `heavy`). In this project these are treated strictly as **"advisory insight — not a safety feature"**:

- Hazard flags are **display-only**. They are shown for illustration and are **not** wired to any actuation, force limit, or refusal behavior.
- Hazard flags are **not distilled into the student** and are **never in the ≤125 ms control loop**. Only `grip` and `force` are learned by the on-device reflex.
- The system **does not detect, prevent, or mitigate injury, burns, cuts, drops, or any other harm**, and must not be relied upon to do so.
- A VLM can hallucinate a hazard (or miss one). Advisory output can be wrong.

Nothing in this repository should be read as a claim that ayaka-hand keeps a user or bystander safe.

## Known limitations

- **🔴 Inference server has no auth and open CORS** (`factory/serve/server.py`: `allow_origins=["*"]`, no auth on any endpoint). This is a **pre-public-deployment blocker** — it must be gated (a shared-secret header is the intended fix) before the server is exposed beyond localhost/trusted tunnel. Do not deploy it as-is.
- **Secrets** are read from a `.env` at the repo root (see `.env.example`). Never commit real keys.
- **Browser WASM path**: the ONNX-Runtime loader has a separate code path for Apple WebKit (Safari uses the plain WASM build; other browsers use the asyncify build). Both are bundled, but Safari/iOS is the most fragile target — verify there first.
- **Detector weights** for the server (LLMDet) are downloaded at runtime, not vendored.
- Training/labeling actually ran on a local NVIDIA RTX 5090 (cu128). The pipeline's vLLM `guided_json` cortex dialect has been **validated end-to-end on real AMD hardware** — an AMD ROCm instance (RDNA3 / `gfx1100`, 48 GB, ROCm 7.2, vLLM 0.16, PyTorch 2.9 + HIP) served Qwen3-VL (the **8B** variant — a proof of the serving path, not the full 32B Cortex) and returned schema-valid grasp JSON at ~4 s/request. This was **not** an MI300X (that class of card remains the larger-scale deployment target, not yet exercised); the validation confirms the ROCm serving path, not the MI300X-specific throughput. CPU inference works but is slow.

## License

Code and documentation are licensed under the **[Apache License 2.0](LICENSE)**.

The license covers this repository's contents only. The training images themselves are **not distributed here** — they derive from license-filtered COCO 2017 source photos (CC BY 2.0 / CC BY-SA 2.0) and are reproducible via the shipped manifests and recipe, with per-image attribution preserved (see the training-data section above).

## Acknowledgements

- **COCO 2017** dataset (license-filtered crops) — [cocodataset.org](https://cocodataset.org/).
- **Qwen3-VL-32B-Instruct** (Apache-2.0) — grasp-label teacher, via OpenRouter.
- **timm** `mobilenetv2_100.ra_in1k` (Apache-2.0) — student backbone.
- **D-FINE** / **RT-DETRv2** ONNX exports by `onnx-community` (Apache-2.0) — in-browser detectors.
- **ONNX Runtime Web** (MIT) — client-side inference.
