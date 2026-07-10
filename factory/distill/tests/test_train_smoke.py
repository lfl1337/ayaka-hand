import json

from distill.train import main


def test_two_epoch_cpu_smoke(crops_root, tmp_path):
    out = tmp_path / "run"
    main(["--data-root", str(crops_root), "--train-labels", "labels-test.jsonl",
          "--val-labels", "labels-test.jsonl", "--out-dir", str(out),
          "--epochs", "2", "--batch-size", "4", "--workers", "0", "--device", "cpu"])
    hist = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert len(hist) == 2
    assert {"epoch", "train_loss", "val_grip_acc", "val_grip_f1", "val_force_acc"} <= set(hist[0])
    assert (out / "best.pt").exists()


def test_checkpoint_carries_arch(crops_root, tmp_path):
    out = tmp_path / "run"
    main(["--data-root", str(crops_root), "--train-labels", "labels-test.jsonl",
          "--val-labels", "labels-test.jsonl", "--out-dir", str(out),
          "--epochs", "1", "--batch-size", "4", "--workers", "0", "--device", "cpu"])
    import torch
    assert torch.load(out / "best.pt", map_location="cpu")["arch"] == "tv"
