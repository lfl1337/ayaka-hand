"""AMD ROCm vLLM guided_json probe — mirrors factory/probe/cortex_call.py's 'vllm' dialect.
Runs against the local vLLM OpenAI server (Qwen3-VL-8B) on gfx1100 / ROCm.
Generates a synthetic 'bottle' test image locally (no external fetch), then measures
end-to-end guided-JSON latency for the grasp-planning schema."""
import base64, json, time, io
from openai import OpenAI
from PIL import Image, ImageDraw

# --- schema v1, inlined verbatim from factory/schema/grasp_schema_v1.json ---
SCHEMA = {
    "type": "object",
    "properties": {
        "object_label":   {"type": "string"},
        "grip":           {"type": "string", "enum": ["power", "lateral", "pinch", "tripod", "no_grasp"]},
        "force":          {"type": "string", "enum": ["delicate", "firm"]},
        "contact_region": {"type": "string", "enum": ["handle", "body", "rim", "edge", "top", "unknown"]},
        "contact_point":  {"type": "object",
                           "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                           "required": ["x", "y"], "additionalProperties": False},
        "hazards":        {"type": "array",
                           "items": {"type": "string", "enum": ["hot", "sharp", "fragile", "full", "heavy", "slippery"]}},
        "rationale":      {"type": "string"},
    },
    "required": ["object_label", "grip", "force", "contact_region", "contact_point", "hazards", "rationale"],
    "additionalProperties": False,
}

PROMPT = (
    "You are the grasp-planning cortex of a SINGLE prosthetic hand (palm-sized, one-handed grasps only). "
    "Look at the main graspable object. Return ONLY JSON matching the schema.\n"
    "Grip rules: power = cylinders, bottles, mugs, larger handles, whole-hand wrap; "
    "lateral = thin flat items held against the side of the index finger (utensils, cards, phones, keys); "
    "pinch = small thin items held between thumb and index fingertip; "
    "tripod = roundish palm-sized objects held with thumb and two fingers (apples, oranges, balls, eggs, small jars); "
    "no_grasp = nothing is safely graspable ONE-HANDED.\n"
    "force=delicate for fragile, light, hot or full containers, else firm. "
    "contact_region/contact_point say where the hand should touch, contact_point normalized x,y in [0,1]. "
    "hazards only if clearly visible, else empty. rationale <= 30 words.\n"
    "JSON schema (follow it exactly, no extra keys):\n"
)

def synth_bottle(path):
    """A crude but unambiguous water-bottle silhouette: tall blue-capped clear cylinder on white."""
    W, H = 512, 512
    img = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(img)
    # body (tall rounded rectangle, light cyan = clear plastic w/ water)
    d.rounded_rectangle([200, 150, 312, 470], radius=30, fill=(205, 225, 235), outline=(150, 175, 185), width=3)
    # neck
    d.rectangle([238, 110, 274, 155], fill=(205, 225, 235), outline=(150, 175, 185), width=2)
    # blue cap
    d.rounded_rectangle([234, 80, 278, 115], radius=6, fill=(30, 70, 170))
    # water line
    d.line([204, 250, 308, 250], fill=(120, 160, 180), width=2)
    img.save(path)
    return path

def main():
    img_path = synth_bottle("/workspace/synth_bottle.png")
    b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")
    model = "Qwen/Qwen3-VL-8B-Instruct"
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": PROMPT + json.dumps(SCHEMA)}]}]

    # warm-up (1) then timed runs (3) — the 'vllm' dialect: extra_body guided_json
    lat = []
    out = None
    for i in range(4):
        t0 = time.time()
        r = client.chat.completions.create(
            model=model, messages=messages, max_tokens=400, temperature=0.2,
            extra_body={"guided_json": SCHEMA})
        dt = time.time() - t0
        content = r.choices[0].message.content or ""
        out = json.loads(content)
        ct = r.usage.completion_tokens if r.usage else None
        if i > 0:  # skip warm-up
            lat.append((dt, ct))
        print(f"[run {i}{' warmup' if i==0 else ''}] {dt*1000:.0f} ms, {ct} tok, grip={out.get('grip')} force={out.get('force')}")

    print("\n=== RESULT (guided_json parsed, schema-valid) ===")
    print(json.dumps(out, indent=2))
    avg_ms = sum(d for d, _ in lat) / len(lat) * 1000
    avg_tok = sum(c for _, c in lat) / len(lat)
    tps = avg_tok / (avg_ms / 1000)
    print(f"\n=== LATENCY (3 timed runs, guided_json) ===")
    print(f"avg {avg_ms:.0f} ms/req | {avg_tok:.0f} completion tok/req | {tps:.1f} tok/s")
    print("device: AMD ROCm (gfx1100, RDNA3, 48GB) | vLLM 0.16.1 | torch 2.9.1+hip7.2 | Qwen3-VL-8B-Instruct")

if __name__ == "__main__":
    main()
