"""cortex_call.py — shared cortex invocation for probe/cache/labeling scripts.

One entry point, four structured-output dialects:
  fireworks : response_format={"type": "json_object", "schema": SCHEMA}
  openai    : response_format={"type": "json_schema", ...}
  vllm      : extra_body={"guided_json": SCHEMA}           <- target path on our MI300X
  prefill   : assistant-prefill '{"object_label":' — bypasses inline thinking of
              reasoning models (Kimi K2.6 on Fireworks); JSON = prefill + completion.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # openai is only a type hint here — keep it out of the runtime import graph
    from openai import OpenAI  # so the inference server can reuse PROMPT/SCHEMA/validate without openai installed

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "schema" / "grasp_schema_v1.json").read_text())

PREFILL = '{"object_label":'

PROMPT_VERSION = "1.1"

PROMPT = (
    "You are the grasp-planning cortex of a SINGLE prosthetic hand (palm-sized, one-handed grasps only). "
    "Look at the main graspable object. Return ONLY JSON matching the schema.\n"
    "Grip rules: power = cylinders, bottles, mugs, larger handles, whole-hand wrap; "
    "lateral = thin flat items held against the side of the index finger (utensils, cards, phones, keys); "
    "pinch = small thin items held between thumb and index fingertip; "
    "tripod = roundish palm-sized objects held with thumb and two fingers (apples, oranges, balls, eggs, small jars); "
    "no_grasp = nothing is safely graspable ONE-HANDED: living beings, objects too large or heavy for one hand "
    "(laptops, keyboards, monitors, furniture, appliances), fixed/attached objects, or no clear target.\n"
    "force=delicate for fragile, light, hot or full containers, else firm. "
    "contact_region/contact_point say where the hand should touch the object, contact_point normalized "
    "x,y in [0,1] relative to the object. hazards only if clearly visible "
    "(hot/sharp/fragile/full/heavy/slippery), else empty. rationale <= 30 words.\n"
    "JSON schema (follow it exactly, no extra keys):\n"
)

GRIPS = {"power", "lateral", "pinch", "tripod", "no_grasp"}
FORCES = {"delicate", "firm"}
REGIONS = {"handle", "body", "rim", "edge", "top", "unknown"}
HAZARDS = {"hot", "sharp", "fragile", "full", "heavy", "slippery"}


def validate(d: dict) -> list[str]:
    errs = []
    for k in SCHEMA["required"]:
        if k not in d:
            errs.append(f"missing key: {k}")
    if d.get("grip") not in GRIPS:
        errs.append(f"bad grip: {d.get('grip')!r}")
    if d.get("force") not in FORCES:
        errs.append(f"bad force: {d.get('force')!r}")
    if d.get("contact_region") not in REGIONS:
        errs.append(f"bad contact_region: {d.get('contact_region')!r}")
    cp = d.get("contact_point")
    if not (isinstance(cp, dict) and isinstance(cp.get("x"), (int, float)) and isinstance(cp.get("y"), (int, float))):
        errs.append(f"bad contact_point: {cp!r}")
    hz = d.get("hazards")
    if not (isinstance(hz, list) and all(h in HAZARDS for h in hz)):
        errs.append(f"bad hazards: {hz!r}")
    return errs


def extract_json(text: str, require_key: str = "grip") -> dict:
    """Direct parse first; else the LAST balanced {...} block that carries require_key
    (thinking models emit prose first; truncation leaves inner fragments like contact_point)."""
    try:
        d = json.loads(text)
        if require_key in d:
            return d
    except json.JSONDecodeError:
        pass
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for start in range(end, -1, -1):
            if text[start] == "}":
                depth += 1
            elif text[start] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(text[start:end + 1])
                        if require_key in d:
                            return d
                    except json.JSONDecodeError:
                        pass
                    break
        end = text.rfind("}", 0, end)
    raise json.JSONDecodeError("no grasp JSON object found in content", text[:80], 0)


def _messages(image_bytes: bytes, mime: str, dialect: str) -> list[dict]:
    b64 = base64.b64encode(image_bytes).decode()
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
        {"type": "text", "text": PROMPT + json.dumps(SCHEMA)}]}]
    if dialect == "prefill":
        msgs.append({"role": "assistant", "content": PREFILL})
    return msgs


def _kwargs(dialect: str) -> dict:
    if dialect == "fireworks":
        return {"response_format": {"type": "json_object", "schema": SCHEMA}}
    if dialect == "openai":
        return {"response_format": {"type": "json_schema",
                                    "json_schema": {"name": "grasp", "strict": True, "schema": SCHEMA}}}
    if dialect == "vllm":
        return {"extra_body": {"guided_json": SCHEMA}}
    if dialect == "prefill":
        return {}
    raise ValueError(f"unknown dialect: {dialect}")


def call_cortex(client: OpenAI, model: str, image_path: Path, dialect: str,
                max_tokens: int = 1600, temperature: float = 0.2,
                retry_tokens: int = 3000) -> tuple[dict, dict]:
    """Returns (validated_json, meta). Raises on schema violations or transport errors.

    Reasoning models sometimes think despite the prefill (stochastic) and may get
    truncated mid-JSON; the retry with a bigger budget lets the thinking finish so
    extract_json can grab the trailing grasp object.
    """
    mime = "png" if image_path.suffix.lower() == ".png" else "jpeg"
    messages = _messages(image_path.read_bytes(), mime, dialect)
    attempts = [max_tokens] + ([retry_tokens] if retry_tokens > max_tokens else [])
    last_err: Exception | None = None
    for budget in attempts:
        r = client.chat.completions.create(model=model, messages=messages,
                                           max_tokens=budget, temperature=temperature,
                                           **_kwargs(dialect))
        content = r.choices[0].message.content or ""
        finish = r.choices[0].finish_reason
        if dialect == "prefill" and not content.lstrip().startswith("{"):
            content = PREFILL + content       # Modell hat den Prefill fortgesetzt
        try:
            out = extract_json(content)
        except json.JSONDecodeError as e:
            last_err = e
            continue                          # Denken angeschnitten → Retry mit mehr Budget
        errs = validate(out)
        if errs:
            if finish == "length":            # Truncation-Artefakt, kein Modellfehler
                last_err = ValueError("truncated: " + "; ".join(errs))
                continue
            raise ValueError("schema violations: " + "; ".join(errs))
        meta = {"completion_tokens": r.usage.completion_tokens if r.usage else None,
                "finish_reason": finish, "dialect": dialect, "budget": budget}
        return out, meta
    raise last_err if last_err else RuntimeError("call_cortex: no attempt succeeded")
