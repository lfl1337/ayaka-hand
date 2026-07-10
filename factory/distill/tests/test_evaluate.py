import json

import torch

from distill.build_dataset import build_index
from distill.evaluate import baseline_predictions, main
from distill.model import DistillStudent


def test_baseline_uses_class_name(crops_root):
    idx = build_index(crops_root, "labels-test.jsonl")
    gp, fp = baseline_predictions(idx["records"])
    assert len(gp) == len(idx["records"]) == len(fp)
    # fixture: banana -> lookup says power(0); knife -> lateral(1)
    assert gp[0] == 0 and gp[4] == 1


def test_cli_writes_table_and_json(crops_root, tmp_path):
    ckpt = tmp_path / "best.pt"
    torch.save({"model": DistillStudent().state_dict(), "epoch": 1, "val_grip_f1": 0.0}, ckpt)
    md, js = tmp_path / "table.md", tmp_path / "results.json"
    main(["--data-root", str(crops_root), "--val-labels", "labels-test.jsonl",
          "--checkpoint", str(ckpt), "--out-md", str(md), "--out-json", str(js),
          "--device", "cpu", "--workers", "0"])
    data = json.loads(js.read_text(encoding="utf-8"))
    assert {"baseline", "student", "n_val", "per_class_grip_f1"} <= set(data)
    text = md.read_text(encoding="utf-8")
    assert "Lookup" in text and "Student" in text and "macro-F1" in text
