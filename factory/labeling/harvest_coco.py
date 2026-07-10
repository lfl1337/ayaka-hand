#!/usr/bin/env python3
"""harvest_coco.py — license-filtered object crops from COCO 2017 for teacher labeling.

Why COCO (documented deviation detail in docs/data/DATASET.md): ground-truth boxes for
15-20 graspable household classes, per-image license field allows filtering to
shippable licenses. NC and NoDerivs images are EXCLUDED (crops are derivatives).

Usage:
  python harvest_coco.py <instances_json> <split> <out_dir> [max_per_class] [global_cap]
  # split: val2017 | train2017   (images fetched per-file from images.cocodataset.org)

Output:
  <out_dir>/crops/<class>/<imageid>_<annid>.jpg   (longest side 320, q88)
  <out_dir>/manifest-<split>.jsonl                (crop, class, ids, bbox, license, urls)
"""
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image

# COCO 2017 license ids: 1-3 NonCommercial*, 6 NoDerivs -> excluded; 4 Attribution,
# 5 Attribution-ShareAlike, 7 No known copyright restrictions, 8 US Government Work -> ok
LICENSE_WHITELIST = {4, 5, 7, 8}

GRASPABLE = ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
             "apple", "orange", "carrot", "donut", "mouse", "remote", "cell phone",
             "book", "vase", "scissors", "teddy bear", "toothbrush", "sports ball", "frisbee"]
NO_GRASP_ANCHORS = ["keyboard", "laptop", "tv", "microwave"]

MIN_SIDE = 64          # px in the original image
MARGIN = 0.15          # crop margin around the bbox
OUT_LONGEST = 320      # resized crop, longest side


def main() -> None:
    instances_json, split, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    max_per_class = int(sys.argv[4]) if len(sys.argv) > 4 else 900
    global_cap = int(sys.argv[5]) if len(sys.argv) > 5 else 20000

    print(f"loading {instances_json} ...", flush=True)
    coco = json.loads(instances_json.read_text())
    lic_names = {l["id"]: l["name"] for l in coco["licenses"]}
    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    wanted = {cid for cid, name in cat_by_id.items() if name in set(GRASPABLE + NO_GRASP_ANCHORS)}
    images = {im["id"]: im for im in coco["images"] if im.get("license") in LICENSE_WHITELIST}
    print(f"{len(images)} images pass the license filter "
          f"({len(coco['images'])} total)", flush=True)

    per_class: dict[str, list[dict]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in wanted or ann["image_id"] not in images or ann.get("iscrowd"):
            continue
        x, y, w, h = ann["bbox"]
        if min(w, h) < MIN_SIDE:
            continue
        per_class.setdefault(cat_by_id[ann["category_id"]], []).append(ann)

    # deterministic balance: sort by annotation id, take head of each class
    picked: list[dict] = []
    for cls, anns in sorted(per_class.items()):
        cap = max_per_class if cls in GRASPABLE else max(50, max_per_class // 4)
        picked += sorted(anns, key=lambda a: a["id"])[:cap]
    picked = picked[:global_cap]
    by_image: dict[int, list[dict]] = {}
    for ann in picked:
        by_image.setdefault(ann["image_id"], []).append(ann)
    print(f"{len(picked)} crops from {len(by_image)} images", flush=True)

    crops_dir = out_dir / "crops"
    manifest_path = out_dir / f"manifest-{split}.jsonl"
    done_crops = {json.loads(l)["crop"] for l in manifest_path.read_text().splitlines()} \
        if manifest_path.exists() else set()

    client = httpx.Client(timeout=30, follow_redirects=True)

    def process_image(image_id: int) -> list[dict]:
        im_meta = images[image_id]
        url = f"http://images.cocodataset.org/{split}/{im_meta['file_name']}"
        rows = []
        todo = [a for a in by_image[image_id]
                if f"crops/{cat_by_id[a['category_id']]}/{image_id}_{a['id']}.jpg" not in done_crops]
        if not todo:
            return []
        for attempt in range(3):
            try:
                img = Image.open(io.BytesIO(client.get(url).raise_for_status().content)).convert("RGB")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"skip {url}: {e}", flush=True)
                    return []
        for ann in todo:
            cls = cat_by_id[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            mx, my = w * MARGIN, h * MARGIN
            box = (max(0, int(x - mx)), max(0, int(y - my)),
                   min(img.width, int(x + w + mx)), min(img.height, int(y + h + my)))
            crop = img.crop(box)
            scale = OUT_LONGEST / max(crop.size)
            if scale < 1:
                crop = crop.resize((round(crop.width * scale), round(crop.height * scale)))
            rel = f"crops/{cls}/{image_id}_{ann['id']}.jpg"
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            crop.save(dest, "JPEG", quality=88)
            rows.append({"crop": rel, "class": cls, "split": split,
                         "coco_image_id": image_id, "ann_id": ann["id"],
                         "bbox": [round(v, 1) for v in ann["bbox"]],
                         "license_id": im_meta["license"], "license": lic_names[im_meta["license"]],
                         "coco_url": im_meta.get("coco_url", url),
                         "flickr_url": im_meta.get("flickr_url", "")})
        return rows

    crops_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with manifest_path.open("a") as mf, ThreadPoolExecutor(max_workers=12) as pool:
        for rows in pool.map(process_image, list(by_image)):
            for row in rows:
                mf.write(json.dumps(row) + "\n")
                written += 1
            if written and written % 500 == 0:
                mf.flush()
                print(f"{written} crops written ...", flush=True)

    counts: dict[str, int] = {}
    for line in manifest_path.read_text().splitlines():
        counts[json.loads(line)["class"]] = counts.get(json.loads(line)["class"], 0) + 1
    print(f"\nDONE: {written} new crops (manifest total {sum(counts.values())}) -> {out_dir}")
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:14s} {n}")


if __name__ == "__main__":
    main()
