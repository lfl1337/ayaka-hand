"""best.pt -> student.onnx (opset 17, dynamic batch) with built-in ORT parity check.
Label orders travel inside the file as metadata — the consumer must never guess them."""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from distill.model import DistillStudent
from distill.schema import FORCES, GRIPS


def export(checkpoint: Path, out: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = DistillStudent(arch=ckpt.get("arch", "tv"))
    model.load_state_dict(ckpt["model"])
    model.eval()
    dummy = torch.randn(1, 3, 160, 160)
    # Pinning dynamo=False (below) revives the legacy TorchScript exporter, which is
    # deliberate but noisy: torch>=2.9 fires DeprecationWarnings for it. Silence exactly
    # those here so the intentional legacy path doesn't pollute output — the pin, not an
    # oversight, is what triggers them.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            model, (dummy,), str(out), opset_version=17,
            input_names=["image"], output_names=["grip_logits", "force_logits"],
            dynamic_axes={"image": {0: "batch"}, "grip_logits": {0: "batch"}, "force_logits": {0: "batch"}},
            dynamo=False,  # torch>=2.9 defaults to the dynamo exporter, which ignores dynamic_axes semantics — pin the legacy path
        )
    m = onnx.load(str(out))
    for key, val in (("grips", json.dumps(GRIPS)), ("forces", json.dumps(FORCES))):
        p = m.metadata_props.add()
        p.key, p.value = key, val
    onnx.save(m, str(out))

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.random.rand(2, 3, 160, 160).astype(np.float32)
    og, of = sess.run(None, {"image": x})
    with torch.no_grad():
        tg, tf = model(torch.from_numpy(x))
    dg = float(np.abs(og - tg.numpy()).max())
    df = float(np.abs(of - tf.numpy()).max())
    if max(dg, df) > 1e-3:
        raise SystemExit(f"ONNX parity FAILED: grip Δ={dg:.2e}, force Δ={df:.2e}")
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB), parity Δ grip={dg:.2e} force={df:.2e}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
