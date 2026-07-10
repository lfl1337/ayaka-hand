#!/usr/bin/env python3
"""day1_probe.py — one structured grasp call against any OpenAI-compatible VLM endpoint.

Usage:
  python day1_probe.py <base_url> <model> <image_path> [dialect ...]
  dialects default: vllm openai fireworks prefill (first that works wins)
  env: API_KEY (fallback FIREWORKS_API_KEY, else "none" for local vLLM)
"""
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from cortex_call import call_cortex


def main() -> None:
    base_url, model, image_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    dialects = sys.argv[4:] or ["vllm", "openai", "fireworks", "prefill"]
    api_key = os.environ.get("API_KEY") or os.environ.get("FIREWORKS_API_KEY") or "none"
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300)

    last_err: Exception | None = None
    for dialect in dialects:
        try:
            t0 = time.perf_counter()
            out, meta = call_cortex(client, model, image_path, dialect)
            dt = time.perf_counter() - t0
            print(json.dumps(out, indent=2))
            toks = meta["completion_tokens"] or 0
            rate = f" tok/s={toks / dt:.1f}" if toks else ""
            print(f"\n[{image_path.name}] dialect={dialect} finish={meta['finish_reason']} "
                  f"latency={dt:.2f}s completion_tokens={toks}{rate}")
            print("schema: VALID")
            return
        except Exception as e:
            last_err = e
            print(f"[{dialect}] failed: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
    raise SystemExit(f"all dialects failed; last error: {last_err}")


if __name__ == "__main__":
    main()
