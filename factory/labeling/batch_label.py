#!/usr/bin/env python3
"""batch_label.py — teacher labeling of harvested crops (async, resumable).

Reads a harvest manifest (jsonl), labels each crop via the cortex, appends results to
an output jsonl. Re-running skips already-labeled crops (resume after crash/abort).

Usage:
  python batch_label.py <base_url> <model> <data_dir> <manifest.jsonl> <out.jsonl> \
                        [dialect] [concurrency] [limit]
  env: API_KEY (fallback OPENROUTER_API_KEY, FIREWORKS_API_KEY, else "none")
"""
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probe"))
from cortex_call import PREFILL, PROMPT_VERSION, _kwargs, _messages, extract_json, validate  # noqa: E402


async def label_one(client: AsyncOpenAI, model: str, dialect: str, img: Path) -> tuple[dict, dict]:
    messages = _messages(img.read_bytes(), "jpeg", dialect)
    last_err: Exception | None = None
    for budget in (1600, 3000):
        r = await client.chat.completions.create(model=model, messages=messages,
                                                 max_tokens=budget, temperature=0.2,
                                                 **_kwargs(dialect))
        content = r.choices[0].message.content or ""
        if dialect == "prefill" and not content.lstrip().startswith("{"):
            content = PREFILL + content
        try:
            out = extract_json(content)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        errs = validate(out)
        if errs:
            if r.choices[0].finish_reason == "length":
                last_err = ValueError("truncated: " + "; ".join(errs))
                continue
            raise ValueError("schema violations: " + "; ".join(errs))
        return out, {"completion_tokens": r.usage.completion_tokens if r.usage else None}
    raise last_err if last_err else RuntimeError("no attempt succeeded")


async def main() -> None:
    base_url, model, data_dir, manifest, out = \
        sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
    dialect = sys.argv[6] if len(sys.argv) > 6 else "openai"
    concurrency = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    limit = int(sys.argv[8]) if len(sys.argv) > 8 else 0

    api_key = (os.environ.get("API_KEY") or os.environ.get("OPENROUTER_API_KEY")
               or os.environ.get("FIREWORKS_API_KEY") or "none")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=300)

    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    done = {json.loads(l)["crop"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in rows if r["crop"] not in done]
    if limit:
        random.seed(11)                      # stratified-ish: shuffle then cut
        random.shuffle(todo)
        todo = todo[:limit]
    print(f"{len(rows)} in manifest, {len(done)} done, labeling {len(todo)} "
          f"(dialect={dialect}, conc={concurrency})", flush=True)

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    stats = {"ok": 0, "err": 0, "tokens": 0}
    t0 = time.perf_counter()

    async def work(row: dict) -> None:
        async with sem:
            for attempt in range(2):
                try:
                    label, meta = await label_one(client, model, dialect, data_dir / row["crop"])
                    async with lock:
                        with out.open("a") as f:
                            f.write(json.dumps({"crop": row["crop"], "class": row["class"],
                                                "teacher": label, "meta": meta,
                                                "prompt_version": PROMPT_VERSION}) + "\n")
                        stats["ok"] += 1
                        stats["tokens"] += meta["completion_tokens"] or 0
                        if stats["ok"] % 25 == 0:
                            rate = stats["ok"] / (time.perf_counter() - t0)
                            print(f"  {stats['ok']}/{len(todo)} ok ({rate:.1f}/s, "
                                  f"{stats['err']} err)", flush=True)
                    return
                except Exception as e:
                    if attempt == 1:
                        async with lock:
                            stats["err"] += 1
                            with out.open("a") as f:
                                f.write(json.dumps({"crop": row["crop"], "class": row["class"],
                                                    "error": f"{type(e).__name__}: {str(e)[:160]}"}) + "\n")
                    else:
                        await asyncio.sleep(2)

    out.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(*[work(r) for r in todo])

    dt = time.perf_counter() - t0
    print(f"\nDONE {stats['ok']} ok / {stats['err']} err in {dt:.0f}s "
          f"({stats['ok'] / dt if dt else 0:.2f} frames/s)")

    # label distribution over EVERYTHING labeled so far
    grips: dict[str, int] = {}
    forces: dict[str, int] = {}
    hazard_n = 0
    total = 0
    for line in out.read_text().splitlines():
        d = json.loads(line)
        if "teacher" not in d:
            continue
        total += 1
        grips[d["teacher"]["grip"]] = grips.get(d["teacher"]["grip"], 0) + 1
        forces[d["teacher"]["force"]] = forces.get(d["teacher"]["force"], 0) + 1
        hazard_n += bool(d["teacher"]["hazards"])
    print(f"grip distribution ({total}): {json.dumps(grips)}")
    print(f"force distribution: {json.dumps(forces)}  | with hazards: {hazard_n}")


if __name__ == "__main__":
    asyncio.run(main())
