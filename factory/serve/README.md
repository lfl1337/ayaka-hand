# ayaka-hand inference server

Optional FastAPI service that powers the live **Cortex "second opinion"** panel in the
browser demo. The browser app runs fully client-side without it; you only need this server
if you want the VLM second-opinion path (`?remote=1`).

**Detector:** open-vocabulary **LLMDet** (`iSEE-Laboratory/llmdet_base`, via `transformers`)
constrained to our class list (`classes.txt`), so off-list COCO confusions are structurally
impossible. **RT-DETRv2** (ONNX) stays as a selectable fallback (`?det=rtdetr`, or if LLMDet
fails to load at startup). Runs on GPU where available (`onnxruntime-directml` on Windows,
CUDA/CPU otherwise).

**Endpoints:** `GET /health`, `POST /infer`, `POST /snap` (broadcast + async Cortex),
`POST /cortex-toggle`, `GET /events` (SSE). Detection JSON matches the web `Detection`
contract (absolute-pixel boxes).

> ### ⚠ Security — not production-ready
> This server currently sets **CORS `allow_origins=["*"]`** and has **no authentication on any
> endpoint**. It is safe only for `127.0.0.1` or a trusted tunnel. **Do not expose it to a
> public network as-is.** Gating it (e.g. a shared-secret header) is a required pre-deployment
> fix — see the root README's *Known limitations*.

## Setup

The Cortex second opinion needs an [OpenRouter](https://openrouter.ai/) key. Copy the template
and fill it in:

```
cp ../../.env.example ../../.env     # then set OPENROUTER_API_KEY in the repo-root .env
```

## Run

```
cd factory/serve
uv sync
uv run uvicorn server:app --host 127.0.0.1 --port 27461
curl -s localhost:27461/health
```

On first start the LLMDet weights are downloaded from Hugging Face. Cortex is reported as
`available` only if a valid `OPENROUTER_API_KEY` is present; otherwise detection still works and
the second-opinion path is disabled.
