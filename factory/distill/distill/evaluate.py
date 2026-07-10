"""Before/after: lookup-table strawman (2017 baseline, gets the GROUND-TRUTH class
name = perfect-detector assumption) vs the distilled student (pixels only).
Output feeds deck + README — numbers must be reproducible from val labels alone."""
import argparse
import json
from pathlib import Path

import torch

from distill.build_dataset import build_index
from distill.data import make_loader
from distill.metrics import accuracy, macro_f1, per_class_f1
from distill.model import DistillStudent, count_params
from distill.schema import FORCE_TO_IDX, FORCES, GRIP_TO_IDX, GRIPS, lookup_baseline


def baseline_predictions(records: list[dict]) -> tuple[list[int], list[int]]:
    grips, forces = [], []
    for r in records:
        g, f = lookup_baseline(r["class"])
        grips.append(GRIP_TO_IDX[g])
        forces.append(FORCE_TO_IDX[f])
    return grips, forces


@torch.no_grad()
def student_predictions(model, loader, device: str) -> tuple[list[int], list[int]]:
    model.eval()
    gp, fp = [], []
    for x, _, _ in loader:
        lg, lf = model(x.to(device))
        gp += lg.argmax(1).cpu().tolist()
        fp += lf.argmax(1).cpu().tolist()
    return gp, fp


def _stats(gp, fp, gt, ft) -> dict:
    return {"grip_acc": round(accuracy(gp, gt), 4), "grip_f1": round(macro_f1(gp, gt, len(GRIPS)), 4),
            "force_acc": round(accuracy(fp, ft), 4), "force_f1": round(macro_f1(fp, ft, len(FORCES)), 4)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    idx = build_index(args.data_root, args.val_labels)
    recs = idx["records"]
    gt = [r["grip_idx"] for r in recs]
    ft = [r["force_idx"] for r in recs]

    bg, bf = baseline_predictions(recs)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = DistillStudent(arch=ckpt.get("arch", "tv")).to(args.device)
    model.load_state_dict(ckpt["model"])
    loader = make_loader(args.data_root, recs, False, args.batch_size, args.workers)
    sg, sf = student_predictions(model, loader, args.device)

    res = {"n_val": len(recs), "baseline": _stats(bg, bf, gt, ft), "student": _stats(sg, sf, gt, ft),
           "per_class_grip_f1": {GRIPS[c]: {"baseline": round(per_class_f1(bg, gt, len(GRIPS))[c], 4),
                                            "student": round(per_class_f1(sg, gt, len(GRIPS))[c], 4)}
                                 for c in range(len(GRIPS))}}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=1), encoding="utf-8")

    b, s = res["baseline"], res["student"]
    params_m = f"{count_params(model) / 1e6:.1f}".replace(".", ",")   # Param-Label ehrlich aus dem geladenen Modell, deutsche Dezimalschreibweise
    md = [
        "# Distillation: Vorher/Nachher (Val n={})".format(res["n_val"]),
        "",
        f"| Metrik | Lookup-Baseline (2017er Strohmann, perfekter Detektor) | Student ({params_m}M CNN, nur Pixel) |",
        "|---|---|---|",
        f"| Grip Accuracy | {b['grip_acc']:.1%} | {s['grip_acc']:.1%} |",
        f"| Grip macro-F1 | {b['grip_f1']:.3f} | {s['grip_f1']:.3f} |",
        f"| Force Accuracy | {b['force_acc']:.1%} | {s['force_acc']:.1%} |",
        "",
        "| Grip-Klasse | Baseline F1 | Student F1 |",
        "|---|---|---|",
    ] + [f"| {g} | {v['baseline']:.3f} | {v['student']:.3f} |"
         for g, v in res["per_class_grip_f1"].items()] + [
        "",
        "_Referenz = Teacher-Labels (Qwen3-VL-32B, Prompt v1.1). Baseline bekommt den"
        " Ground-Truth-Klassennamen (stärkstmöglicher Strohmann); der Student sieht nur Pixel._",
    ]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(res["baseline"]), json.dumps(res["student"]))


if __name__ == "__main__":
    main()
