"""Hard-label distillation of the VLM teacher into the edge student.

Loss = weighted CE(grip) + 0.5 * weighted CE(force); inverse-frequency weights
from the builder (grip classes are imbalanced). bf16 autocast on CUDA, AdamW,
cosine schedule, best checkpoint by val grip macro-F1.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from distill.build_dataset import build_index
from distill.data import make_loader
from distill.metrics import accuracy, macro_f1
from distill.model import DistillStudent
from distill.schema import FORCES, GRIPS

FORCE_LOSS_WEIGHT = 0.5


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device: str) -> dict:
    model.eval()
    gp, gt, fp, ft = [], [], [], []
    for x, g, f in loader:
        lg, lf = model(x.to(device))
        gp += lg.argmax(1).cpu().tolist(); gt += g.tolist()
        fp += lf.argmax(1).cpu().tolist(); ft += f.tolist()
    return {"val_grip_acc": accuracy(gp, gt), "val_grip_f1": macro_f1(gp, gt, len(GRIPS)),
            "val_force_acc": accuracy(fp, ft), "val_force_f1": macro_f1(fp, ft, len(FORCES))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--train-labels", required=True)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch", default="tv", choices=["tv", "timm_mnv2_100", "timm_mnv2_050"])
    ap.add_argument("--pretrained", action="store_true",
                    help="ImageNet-init the backbone (timm archs only)")
    args = ap.parse_args(argv)
    if args.pretrained and args.arch == "tv":
        ap.error("--pretrained requires a timm arch (--arch timm_mnv2_100|timm_mnv2_050); "
                 "torchvision width-0.75 has no ImageNet weights")

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tr_idx = build_index(args.data_root, args.train_labels)
    va_idx = build_index(args.data_root, args.val_labels)
    (args.out_dir / "index-train.json").write_text(json.dumps(tr_idx), encoding="utf-8")
    (args.out_dir / "index-val.json").write_text(json.dumps(va_idx), encoding="utf-8")
    print(f"train={len(tr_idx['records'])} (skipped {tr_idx['skipped']}) "
          f"val={len(va_idx['records'])} (skipped {va_idx['skipped']}) device={args.device}")

    tr = make_loader(args.data_root, tr_idx["records"], True, args.batch_size, args.workers)
    va = make_loader(args.data_root, va_idx["records"], False, args.batch_size, args.workers)

    model = DistillStudent(arch=args.arch, load_pretrained=args.pretrained).to(args.device)
    ce_grip = nn.CrossEntropyLoss(weight=torch.tensor(tr_idx["grip_weights"], device=args.device))
    ce_force = nn.CrossEntropyLoss(weight=torch.tensor(tr_idx["force_weights"], device=args.device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    use_amp = args.device == "cuda"

    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, nb = time.perf_counter(), 0.0, 0
        for x, g, f in tr:
            x, g, f = x.to(args.device), g.to(args.device), f.to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                lg, lf = model(x)
                loss = ce_grip(lg, g) + FORCE_LOSS_WEIGHT * ce_force(lf, f)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item(); nb += 1
        sched.step()
        row = {"epoch": epoch, "train_loss": round(total / max(nb, 1), 4),
               "secs": round(time.perf_counter() - t0, 1),
               **{k: round(v, 4) for k, v in evaluate_model(model, va, args.device).items()}}
        history.append(row)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        print(json.dumps(row))
        if row["val_grip_f1"] > best_f1:
            best_f1 = row["val_grip_f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_grip_f1": best_f1,
                        "arch": args.arch}, args.out_dir / "best.pt")
    print(f"best val grip macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
