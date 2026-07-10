import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

from distill.export_onnx import export
from distill.model import DistillStudent


def test_export_parity_and_metadata(tmp_path):
    ckpt = tmp_path / "best.pt"
    model = DistillStudent()
    torch.save({"model": model.state_dict(), "epoch": 1, "val_grip_f1": 0.0}, ckpt)
    out = tmp_path / "student.onnx"
    export(ckpt, out)

    m = onnx.load(str(out))
    meta = {p.key: p.value for p in m.metadata_props}
    assert json.loads(meta["grips"])[0] == "power" and len(json.loads(meta["forces"])) == 2

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.random.rand(3, 3, 160, 160).astype(np.float32)          # dynamic batch = 3
    og, of = sess.run(None, {"image": x})
    model.eval()
    with torch.no_grad():
        tg, tf = model(torch.from_numpy(x))
    assert og.shape == (3, 5) and of.shape == (3, 2)
    assert np.abs(og - tg.numpy()).max() < 1e-3
    assert np.abs(of - tf.numpy()).max() < 1e-3
