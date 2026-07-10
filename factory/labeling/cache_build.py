#!/usr/bin/env python3
"""cache_build.py — label every image in a dir, write cache.json keyed by sha256.

The web demo ships this cache as its fallback: dead endpoint degrades to cached
cortex responses instead of a 404 (honesty badge handled client-side).

Usage:
  python cache_build.py <base_url> <model> <image_dir> <out_json> [dialect]
  env: API_KEY (fallback FIREWORKS_API_KEY, else "none" for local vLLM)
"""
import hashlib
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probe"))
from cortex_call import call_cortex  # noqa: E402


def main() -> None:
    base_url, model, image_dir, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    dialect = sys.argv[5] if len(sys.argv) > 5 else "vllm"
    api_key = os.environ.get("API_KEY") or os.environ.get("FIREWORKS_API_KEY") or "none"
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300)

    cache: dict[str, dict] = json.loads(out.read_text()) if out.exists() else {}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    fresh = 0
    for img in images:
        key = hashlib.sha256(img.read_bytes()).hexdigest()
        if key in cache:
            print(f"cached  {img.name}")
            continue
        response, _ = call_cortex(client, model, img, dialect)
        cache[key] = response
        fresh += 1
        print(f"labeled {img.name} -> {response['grip']}/{response['force']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, indent=1))
    print(f"{len(cache)} entries ({fresh} new) -> {out}")


if __name__ == "__main__":
    main()
