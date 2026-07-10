# factory/serve/server.py
"""Remote open-vocabulary inference server for the ayaka-hand studio demo.

Primary detector: LLMDet (open-vocabulary, transformers) constrained to OUR
class list (classes.txt) — so off-list COCO confusions (white mug -> "toilet",
headphone case -> "scissors") are structurally impossible: the model can only
emit one of our prompt phrases.

RT-DETRv2-ONNX stays as a selectable fallback. Backend select:
    env DETECTOR=ovd|rtdetr (default ovd)  +  per-request ?det=rtdetr on /infer,/snap
If the LLMDet load fails at startup we log and fall back to rtdetr — never crash.

Output JSON matches the web `Detection` contract exactly:
    {label: str, score: float, box: {xmin, ymin, xmax, ymax}}  # absolute px
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ayaka-serve")

MODEL_DIR = Path(__file__).parent / "model"
ONNX_PATH = MODEL_DIR / "onnx" / "model_quantized.onnx"
CONFIG_PATH = MODEL_DIR / "config.json"
PREPROC_PATH = MODEL_DIR / "preprocessor_config.json"
CLASSES_PATH = Path(__file__).parent / "classes.txt"

SCORE_THRESHOLD = 0.4   # rtdetr focal head — matches web CONFIG.detector.scoreThreshold
BOX_THRESHOLD = 0.25    # ovd/LLMDet: min box confidence
TEXT_THRESHOLD = 0.2    # ovd/LLMDet: min phrase-match score (passed if the installed API accepts it)

OVD_MODEL_ID = "iSEE-Laboratory/llmdet_base"
DETECTOR = os.environ.get("DETECTOR", "ovd").strip().lower()  # 'ovd' | 'rtdetr'

PREVIEW_LONGEST = 640   # SSE preview: re-encode longest side to this
PREVIEW_QUALITY = 80


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_classes() -> list[str]:
    """One detection prompt phrase per line; blanks and #-comments skipped."""
    if not CLASSES_PATH.is_file():
        return []
    out: list[str] = []
    for line in CLASSES_PATH.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# RT-DETRv2 backend (ONNX, numpy+PIL pre/post) — the COCO-80 fallback
# ---------------------------------------------------------------------------


class RtdetrBackend:
    name = "rtdetr (onnx)"

    def __init__(self) -> None:
        import onnxruntime as ort  # lazy: keep module import-safe where ORT/files are absent

        self._ort = ort
        cfg = _load_json(CONFIG_PATH)
        pp = _load_json(PREPROC_PATH)
        self.id2label: dict[int, str] = {int(k): v for k, v in cfg["id2label"].items()}

        size = pp.get("size", {"height": 640, "width": 640})
        self.in_h = int(size["height"])
        self.in_w = int(size["width"])
        self.do_resize = bool(pp.get("do_resize", True))
        self.do_rescale = bool(pp.get("do_rescale", True))
        self.rescale = float(pp.get("rescale_factor", 1.0 / 255.0))
        self.do_normalize = bool(pp.get("do_normalize", False))
        self.mean = np.array(pp.get("image_mean", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(pp.get("image_std", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, 1, 3)
        try:
            self.resample = Image.Resampling(int(pp.get("resample", 2)))  # 2 == BILINEAR
        except ValueError:
            self.resample = Image.Resampling.BILINEAR

        self.session = self._build_session()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.provider = self.session.get_providers()[0]
        self.device = self.provider

        log.info("rtdetr ONNX loaded: %s", ONNX_PATH)
        log.info("rtdetr available EPs: %s", ort.get_available_providers())
        log.info("rtdetr active EP: %s", self.provider)
        log.info("rtdetr: %d classes, threshold=%.2f, input=%dx%d normalize=%s",
                 len(self.id2label), SCORE_THRESHOLD, self.in_w, self.in_h, self.do_normalize)

    def _build_session(self) -> Any:
        ort = self._ort
        available = ort.get_available_providers()
        preferred: list[str] = []
        if "DmlExecutionProvider" in available:
            preferred.append("DmlExecutionProvider")
        if "CUDAExecutionProvider" in available and "DmlExecutionProvider" not in available:
            preferred.append("CUDAExecutionProvider")
        preferred.append("CPUExecutionProvider")
        return ort.InferenceSession(str(ONNX_PATH), providers=preferred)

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """PIL image -> float32 NCHW [1,3,H,W]. Resize (no aspect preserve),
        rescale 1/255, optional normalize — exactly per preprocessor_config.json."""
        rgb = img.convert("RGB")
        if self.do_resize:
            rgb = rgb.resize((self.in_w, self.in_h), self.resample)
        arr = np.asarray(rgb, dtype=np.float32)  # HWC, 0..255
        if self.do_rescale:
            arr = arr * self.rescale
        if self.do_normalize:
            arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return np.ascontiguousarray(arr[None, ...], dtype=np.float32)  # NCHW

    def _split_outputs(self, outs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        logits = pred_boxes = None
        for name, arr in zip(self.output_names, outs):
            if name == "logits":
                logits = arr
            elif name == "pred_boxes":
                pred_boxes = arr
        if logits is None or pred_boxes is None:  # fallback: bind by shape
            for arr in outs:
                if arr.ndim == 3 and arr.shape[-1] == 4:
                    pred_boxes = arr
                elif arr.ndim == 3:
                    logits = arr
        if logits is None or pred_boxes is None:
            raise RuntimeError(f"cannot locate logits/pred_boxes in outputs {self.output_names}")
        return logits, pred_boxes

    def postprocess(self, logits: np.ndarray, pred_boxes: np.ndarray,
                    orig_w: int, orig_h: int) -> list[dict[str, Any]]:
        """RT-DETRv2 focal-loss head: sigmoid, per-query max class, threshold,
        cxcywh(normalized) -> xyxy(absolute px in the ORIGINAL image)."""
        logit = logits[0]          # [num_queries, num_classes]
        boxes = pred_boxes[0]      # [num_queries, 4] cxcywh normalized 0..1
        scores_all = _sigmoid(logit)
        labels = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(scores_all.shape[0]), labels]
        keep = np.nonzero(scores > SCORE_THRESHOLD)[0]

        dets: list[dict[str, Any]] = []
        for i in keep:
            cx, cy, w, h = boxes[i]
            xmin = float((cx - w / 2.0) * orig_w)
            ymin = float((cy - h / 2.0) * orig_h)
            xmax = float((cx + w / 2.0) * orig_w)
            ymax = float((cy + h / 2.0) * orig_h)
            dets.append({
                "label": self.id2label.get(int(labels[i]), f"class_{int(labels[i])}"),
                "score": float(scores[i]),
                "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            })
        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets

    def infer(self, img: Image.Image) -> list[dict[str, Any]]:
        orig_w, orig_h = img.size
        tensor = self.preprocess(img)
        outs = self.session.run(self.output_names, {self.input_name: tensor})
        logits, pred_boxes = self._split_outputs(outs)
        return self.postprocess(logits, pred_boxes, orig_w, orig_h)


# ---------------------------------------------------------------------------
# LLMDet open-vocabulary backend (transformers) — the primary detector
# ---------------------------------------------------------------------------


class OvdBackend:
    name = "llmdet_base (ovd)"

    def __init__(self) -> None:
        import torch  # lazy: only imported when DETECTOR=ovd
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.classes = _load_classes()
        if not self.classes:
            raise RuntimeError(f"classes.txt empty or missing at {CLASSES_PATH}")
        # LLMDet/transformers wants a batch of phrase lists: [[ "cup", "bottle", ... ]].
        self.prompt: list[list[str]] = [[c.lower() for c in self.classes]]
        # Canonicalise returned phrase -> our exact class string (client lookup is string-equality).
        self._by_lower: dict[str, str] = {}
        for c in self.classes:
            self._by_lower.setdefault(c.lower(), c)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(OVD_MODEL_ID)
        # fp32 weights: LLMDet/GroundingDINO fp16 inference throws dtype mismatches in its
        # deformable-attention path (model loads fp16 fine but the forward fails). fp32 is
        # reliable and plenty fast on the 5090 (base model).
        self.model = self._load_model().to(self.device).eval()
        self.dtype = "float32"

        log.info("OVD loaded: %s on %s (%s), %d classes",
                 OVD_MODEL_ID, self.device, self.dtype, len(self.classes))

    def _load_model(self) -> Any:
        torch = self._torch
        from transformers import AutoModelForZeroShotObjectDetection

        # `dtype=` is the current kwarg; older transformers used `torch_dtype=`.
        try:
            return AutoModelForZeroShotObjectDetection.from_pretrained(OVD_MODEL_ID, dtype=torch.float32)
        except TypeError:
            return AutoModelForZeroShotObjectDetection.from_pretrained(OVD_MODEL_ID, torch_dtype=torch.float32)

    def _label_text(self, lab: Any) -> str:
        """Return a phrase string for a post-process label (string phrase, or int index)."""
        if isinstance(lab, str):
            return lab
        try:
            idx = int(lab)
        except (TypeError, ValueError):
            return str(lab)
        if 0 <= idx < len(self.classes):
            return self.classes[idx]
        return str(lab)

    def _canon(self, text: str) -> str | None:
        """Snap a returned phrase to exactly one class string, else None (dropped)."""
        t = text.strip().lower()
        if not t:
            return None
        if t in self._by_lower:
            return self._by_lower[t]
        matches = sorted({c for c in self.classes if c.lower() in t or t in c.lower()})
        return matches[0] if len(matches) == 1 else None

    def _post_process(self, outputs: Any, target_sizes: list[tuple[int, int]]) -> Any:
        """Call post_process_grounded_object_detection across transformers API revisions.
        Newer: threshold(+optional text_threshold). Older GroundingDINO: box_threshold+text_threshold."""
        ppg = self.processor.post_process_grounded_object_detection
        for kwargs in (
            {"threshold": BOX_THRESHOLD, "text_threshold": TEXT_THRESHOLD, "target_sizes": target_sizes},
            {"threshold": BOX_THRESHOLD, "target_sizes": target_sizes},
            {"box_threshold": BOX_THRESHOLD, "text_threshold": TEXT_THRESHOLD, "target_sizes": target_sizes},
        ):
            try:
                return ppg(outputs, **kwargs)
            except TypeError:
                continue
        return ppg(outputs, target_sizes=target_sizes)

    def infer(self, img: Image.Image) -> list[dict[str, Any]]:
        torch = self._torch
        rgb = img.convert("RGB")
        orig_w, orig_h = rgb.size
        inputs = self.processor(images=rgb, text=self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self._post_process(outputs, [(orig_h, orig_w)])  # target_sizes = (H, W)
        res = results[0]

        boxes = res["boxes"].detach().cpu().tolist()
        scores = res["scores"].detach().cpu().tolist()
        raw_labels = res.get("text_labels")
        if raw_labels is None:
            raw_labels = res.get("labels")
        labels = [self._label_text(raw_labels[i]) for i in range(len(boxes))]

        dets: list[dict[str, Any]] = []
        for box, score, lab in zip(boxes, scores, labels):
            canon = self._canon(lab)
            if canon is None:  # off-list phrase — drop it (the whole point of OVD)
                log.info("OVD dropped off-list label %r (score %.2f)", lab, float(score))
                continue
            xmin, ymin, xmax, ymax = (float(v) for v in box)
            dets.append({
                "label": canon,
                "score": float(score),
                "box": {
                    "xmin": max(0.0, min(xmin, orig_w)),
                    "ymin": max(0.0, min(ymin, orig_h)),
                    "xmax": max(0.0, min(xmax, orig_w)),
                    "ymax": max(0.0, min(ymax, orig_h)),
                },
            })
        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets


# ---------------------------------------------------------------------------
# Backend init + dispatch (guarded — never crash the server)
# ---------------------------------------------------------------------------

RT: RtdetrBackend | None = None
OVD: OvdBackend | None = None

try:
    RT = RtdetrBackend()
except Exception as exc:  # noqa: BLE001 - rtdetr optional (files/ORT may be absent)
    log.warning("rtdetr backend unavailable: %s", exc)

if DETECTOR == "ovd":
    try:
        OVD = OvdBackend()
    except Exception as exc:  # noqa: BLE001 - LLMDet load failed -> fall back to rtdetr
        log.error("OVD (LLMDet) load failed — falling back to rtdetr: %s", exc)
        OVD = None

if OVD is not None:
    PRIMARY = "ovd"
elif RT is not None:
    PRIMARY = "rtdetr"
else:
    PRIMARY = "none"
    log.error("NO detector backend available — /infer and /snap will return 503")

log.info("detector: primary=%s (env DETECTOR=%s) rt=%s ovd=%s",
         PRIMARY, DETECTOR, RT is not None, OVD is not None)


def _resolve(override: str | None) -> tuple[str, Any]:
    """Pick a backend: per-request override, else PRIMARY, else whatever is loaded."""
    want = (override or PRIMARY).strip().lower()
    if want == "rtdetr" and RT is not None:
        return "rtdetr", RT
    if want == "ovd" and OVD is not None:
        return "ovd", OVD
    if OVD is not None:
        return "ovd", OVD
    if RT is not None:
        return "rtdetr", RT
    return "none", None


def infer_sync(img: Image.Image, override: str | None = None
               ) -> tuple[list[dict[str, Any]], float, tuple[int, int], str]:
    """Full inference wall time (pre + run + post) in ms, plus the backend used."""
    orig_w, orig_h = img.size
    name, backend = _resolve(override)
    if backend is None:
        raise RuntimeError("no detector backend available")
    t0 = time.perf_counter()
    dets = backend.infer(img)
    ms = (time.perf_counter() - t0) * 1000.0
    return dets, ms, (orig_w, orig_h), name


def _det_override(request: Request) -> str | None:
    v = request.query_params.get("det")
    return v.strip().lower() if v else None


# ---------------------------------------------------------------------------
# Image input parsing (multipart OR raw body)
# ---------------------------------------------------------------------------


async def read_image(request: Request) -> Image.Image:
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        for value in form.values():
            if hasattr(value, "read"):  # UploadFile
                data = await value.read()
                return Image.open(io.BytesIO(data))
        raise ValueError("multipart form contained no file part")
    data = await request.body()
    if not data:
        raise ValueError("empty request body")
    return Image.open(io.BytesIO(data))


def encode_preview(img: Image.Image) -> str:
    """Re-encode longest side to <=640, JPEG q80, base64 (no data: prefix)."""
    rgb = img.convert("RGB")
    longest = max(rgb.size)
    if longest > PREVIEW_LONGEST:
        scale = PREVIEW_LONGEST / float(longest)
        rgb = rgb.resize((max(1, round(rgb.width * scale)),
                          max(1, round(rgb.height * scale))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=PREVIEW_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# SSE fan-out state
# ---------------------------------------------------------------------------

_subscribers: set[asyncio.Queue[str]] = set()
_latest_frame: dict[str, Any] | None = None


async def broadcast(msg: dict[str, Any], *, cache: bool = True) -> None:
    """Fan a message out to every SSE subscriber. Only detector frames are cached
    for replay-on-subscribe (cache=False for cortex events — they must not become
    the 'latest frame' a late viewer sees)."""
    global _latest_frame
    if cache:
        _latest_frame = msg
    payload = json.dumps(msg)
    for q in list(_subscribers):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # slow consumer: drop this frame for them, keep the stream alive


# ---------------------------------------------------------------------------
# Cortex (VLM via OpenRouter) — the slow, smart second opinion on the edge frame
# ---------------------------------------------------------------------------
# PROMPT / SCHEMA / validate / extract_json come from the probe tree (single source
# of truth); openai stays out of that import (guarded by TYPE_CHECKING in cortex_call).

CORTEX_MODEL = "qwen/qwen3-vl-32b-instruct"
CORTEX_URL = "https://openrouter.ai/api/v1/chat/completions"
CORTEX_TIMEOUT = 60.0
CORTEX_MAX_TOKENS = 400
CORTEX_TEMPERATURE = 0.2

_cortex_state = {"on": True}                 # runtime toggle via POST /cortex-toggle
_cortex_tasks: set[asyncio.Task[None]] = set()

try:
    _PROBE_DIR = Path(__file__).resolve().parents[1] / "probe"
    if str(_PROBE_DIR) not in sys.path:
        sys.path.insert(0, str(_PROBE_DIR))
    import cortex_call as cc
except Exception as exc:  # noqa: BLE001 - cortex is optional; the edge reflex must serve regardless
    cc = None  # type: ignore[assignment]
    log.warning("cortex_call import failed — cortex disabled: %s", exc)


def _load_openrouter_key() -> str | None:
    """Env first, else the repo .env two dirs up (factory/serve -> repo root)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key and key.strip():
        return key.strip()
    for candidate in (
        Path(__file__).resolve().parents[2] / ".env",   # repo root (where .env lives)
        Path(__file__).resolve().parents[1] / ".env",   # factory/.env fallback
    ):
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("OPENROUTER_API_KEY="):
                    return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return None


OPENROUTER_KEY = _load_openrouter_key()
CORTEX_AVAILABLE = cc is not None and bool(OPENROUTER_KEY)
log.info("cortex: %s (model=%s)",
         "available" if CORTEX_AVAILABLE else "disabled (no key or import failed)", CORTEX_MODEL)


def _cortex_status() -> str:
    if not CORTEX_AVAILABLE:
        return "disabled"
    return "on" if _cortex_state["on"] else "off"


async def _run_cortex(image_b64: str, ts: float) -> None:
    """Call the OpenRouter VLM on the snapped frame and push its verdict over SSE.
    Fire-and-forget: never blocks /snap, never raises — a cortex failure surfaces as
    an SSE {ok:false} event so the viewer can fall back to the edge reflex visibly."""
    t0 = time.perf_counter()
    payload = {
        "model": CORTEX_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": cc.PROMPT + json.dumps(cc.SCHEMA)},
        ]}],
        "max_tokens": CORTEX_MAX_TOKENS,
        "temperature": CORTEX_TEMPERATURE,
        # exactly cortex_call's 'openai' dialect: strict json_schema structured output
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "grasp", "strict": True, "schema": cc.SCHEMA}},
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=CORTEX_TIMEOUT) as client:
            resp = await client.post(CORTEX_URL, headers=headers, json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
        out = cc.extract_json(content)
        errs = cc.validate(out)
        if errs:
            raise ValueError("schema violations: " + "; ".join(errs))
        ms = (time.perf_counter() - t0) * 1000.0
        await broadcast({"type": "cortex", "ts": ts, "ok": True, "result": out, "ms": round(ms, 2)},
                        cache=False)
        log.info("cortex ok: grip=%s force=%s hazards=%s (%.0f ms)",
                 out.get("grip"), out.get("force"), out.get("hazards"), ms)
    except Exception as exc:  # noqa: BLE001 - fire-and-forget: report as SSE, never crash the loop
        ms = (time.perf_counter() - t0) * 1000.0
        await broadcast({"type": "cortex", "ts": ts, "ok": False,
                         "error": f"{type(exc).__name__}: {str(exc)[:200]}", "ms": round(ms, 2)},
                        cache=False)
        log.warning("cortex failed after %.0f ms: %s", ms, exc)


def _spawn_cortex(image_b64: str, ts: float) -> None:
    """Schedule the cortex call on the running loop and keep a strong ref until it settles."""
    task = asyncio.create_task(_run_cortex(image_b64, ts))
    _cortex_tasks.add(task)
    task.add_done_callback(_cortex_tasks.discard)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ayaka-hand inference", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tunnel-only exposure
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    name, backend = _resolve(None)
    return {
        "ok": backend is not None,
        "detector": backend.name if backend is not None else "none",
        "device": getattr(backend, "device", None) if backend is not None else None,
        "provider": RT.provider if RT is not None else None,   # rtdetr EP (backward-compat)
        "cortex": _cortex_status(),
    }


@app.post("/infer")
async def infer(request: Request) -> JSONResponse:
    try:
        img = await read_image(request)
    except Exception as exc:  # noqa: BLE001 - surface bad input as 400
        return JSONResponse({"error": {"code": "bad_image", "message": str(exc)}}, status_code=400)
    try:
        dets, ms, (w, h), name = await run_in_threadpool(infer_sync, img, _det_override(request))
    except Exception as exc:  # noqa: BLE001 - no backend loaded
        return JSONResponse({"error": {"code": "no_backend", "message": str(exc)}}, status_code=503)
    return JSONResponse({"detections": dets, "ms": round(ms, 2), "size": [w, h], "detector": name})


@app.post("/snap")
async def snap(request: Request) -> JSONResponse:
    try:
        img = await read_image(request)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": {"code": "bad_image", "message": str(exc)}}, status_code=400)
    try:
        dets, ms, (w, h), name = await run_in_threadpool(infer_sync, img, _det_override(request))
    except Exception as exc:  # noqa: BLE001 - no backend loaded
        return JSONResponse({"error": {"code": "no_backend", "message": str(exc)}}, status_code=503)
    b64 = await run_in_threadpool(encode_preview, img)
    ts = time.time()
    frame = {
        "type": "frame",
        "ts": ts,
        "image_jpeg_b64": b64,
        "detections": dets,
        "ms": round(ms, 2),
        "size": [w, h],
        "detector": name,
    }
    await broadcast(frame)
    if CORTEX_AVAILABLE and _cortex_state["on"]:
        _spawn_cortex(b64, ts)               # async second opinion — does NOT block the /snap response
    return JSONResponse({"ok": True, "detections": dets, "ms": round(ms, 2), "size": [w, h], "detector": name})


@app.post("/cortex-toggle")
async def cortex_toggle(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - tolerate an empty/garbage body, default to on
        body = {}
    _cortex_state["on"] = bool(body.get("on", True))
    return JSONResponse({"ok": True, "cortex": _cortex_status()})


@app.get("/events")
async def events() -> StreamingResponse:
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=8)
    _subscribers.add(q)

    async def gen():
        try:
            if _latest_frame is not None:  # push latest immediately on subscribe
                yield f"data: {json.dumps(_latest_frame)}\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # keep-alive comment every 15s
        finally:
            _subscribers.discard(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=27461)
