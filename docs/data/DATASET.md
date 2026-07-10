# Training / labeling dataset — provenance & license

## Source & filter

**COCO 2017** (val2017 + train2017), object crops taken over ground-truth bounding boxes.
Harvest script: `factory/labeling/harvest_coco.py`. A **per-image license filter** is applied —
only COCO license IDs `{4 Attribution, 5 Attribution-ShareAlike, 7 No known copyright restrictions,
8 US Government Work}` are kept. **Excluded:** all NonCommercial variants (1–3) and
Attribution-NoDerivatives (6), because the crops are derivative works. Attribution is satisfied
through the manifest (`flickr_url`, `coco_url`, `license` per crop). Crop parameters: minimum
object side 64 px, 15 % margin, longest side 320 px, JPEG q88.

**Why COCO and not ContactPose / Saudabayev?** Those remain the sources for a later phase
(contact maps and a 35-grasp taxonomy respectively), but are impractical for this slice:
ContactPose is third-person multi-view, terabyte-scale; Saudabayev is 172 GB of MATLAB formats.
COCO crops deliver the same ingredient for teacher labeling (object appearance → grip/force) with
ground-truth boxes and shippable licenses, in minutes rather than days of acquisition time.

## Inventory

| Split | Crops | File |
|---|---|---|
| train2017 | 11,004 | `manifest-train2017.jsonl` |
| val2017 | 613 | `manifest-val2017.jsonl` (basis for eval/test) |

**Actual license breakdown** (both manifests combined): CC BY 2.0 = 7,507; CC BY-SA 2.0 = 4,103;
No known copyright restrictions = 7; total = 11,617.

26 classes: 22 graspable (bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
orange, carrot, donut, mouse, remote, cell phone, book, vase, scissors, teddy bear, toothbrush,
sports ball, frisbee) + 4 **no-grasp anchors** (keyboard, laptop, tv, microwave — too large/heavy
for a one-handed prosthetic grip). Balancing: cap 900/class (anchors 225). Thin classes: sports
ball (48), mouse (87) — accepted.

> The crop **images are not shipped in this repository** (see the root README's *Data & attribution*).
> Only the manifests and the teacher labels are versioned; regenerate the pixels with
> `factory/labeling/harvest_coco.py`.

## Teacher labels

- **Labeler:** `factory/labeling/batch_label.py` (async, resumable, `prompt_version` per line).
- **Prompt v1.1** (in `factory/probe/cortex_call.py`): sharpened after a 200-crop pilot — the
  v1.0 findings were that anchors were mislabeled `power` ("graspable" ≠ "sensibly graspable
  one-handed") and `tripod` was starved (2/200). v1.1 makes the one-hand rule explicit. Effect on
  the same 200 crops: tripod 2→45; keyboard/laptop/tv → mostly `no_grasp`; knife stable `lateral`;
  cup stable `power`.
- **Backup run (OpenRouter, `qwen/qwen3-vl-32b-instruct` = the target model):** the complete set,
  `labels-train2017-backup.jsonl` / `labels-val2017-backup.jsonl`. Purpose: risk insurance +
  prompt validation at scale. Cost ≈ $2–3.

## Archive

`labels-pilot-v1.0.jsonl` (200 crops, Prompt v1.0) — kept as pilot documentation only, not used
for training.
