"""Pre-download + load LLMDet once so the server's eager load is cache-fast.
Also validates the fp16 CUDA path outside uvicorn (clearer failure surface)."""
import time

import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MID = "iSEE-Laboratory/llmdet_base"
t0 = time.time()
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)

AutoProcessor.from_pretrained(MID)
print(f"[prewarm] processor ok ({time.time()-t0:.1f}s)", flush=True)

want_fp16 = torch.cuda.is_available()
dtype = torch.float16 if want_fp16 else torch.float32
try:
    try:
        m = AutoModelForZeroShotObjectDetection.from_pretrained(MID, dtype=dtype)
    except TypeError:
        m = AutoModelForZeroShotObjectDetection.from_pretrained(MID, torch_dtype=dtype)
    used = "float16" if want_fp16 else "float32"
except Exception as exc:  # noqa: BLE001
    print(f"[prewarm] fp16 load failed ({exc}); retrying fp32", flush=True)
    m = AutoModelForZeroShotObjectDetection.from_pretrained(MID, torch_dtype=torch.float32)
    used = "float32"

dev = "cuda" if want_fp16 else "cpu"
m = m.to(dev).eval()
print(f"[prewarm] model ok dtype={used} dev={dev} ({time.time()-t0:.1f}s total)", flush=True)
