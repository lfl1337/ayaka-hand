"""Teacher-JSONL + crops dir -> training index (records, class stats, CE weights).

Rows are validated against Schema v1 enums and the file system; anything invalid
is skipped AND counted — silent data loss corrupts the hero-claim numbers.
"""
import argparse
import json
from pathlib import Path
from typing import Any

from distill.schema import FORCE_TO_IDX, FORCES, GRIP_TO_IDX, GRIPS


def _weights(counts: dict[str, int], order: list[str]) -> list[float]:
    """Inverse-frequency weights, mean-normalized over the PRESENT classes; absent -> 0."""
    inv = [0.0 if counts.get(k, 0) == 0 else 1.0 / counts[k] for k in order]
    present = [v for v in inv if v > 0]
    scale = len(present) / sum(present) if present else 1.0
    return [round(v * scale, 6) for v in inv]


def build_index(data_root: Path, labels_file: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    grip_counts: dict[str, int] = {}
    force_counts: dict[str, int] = {}
    skipped = 0
    with (Path(data_root) / labels_file).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            teacher = row.get("teacher") or {}
            grip, force = teacher.get("grip"), teacher.get("force")
            rel = row.get("crop", "")
            if grip not in GRIP_TO_IDX or force not in FORCE_TO_IDX or not (Path(data_root) / rel).is_file():
                skipped += 1
                continue
            records.append({"path": rel, "class": row.get("class", "unknown"),
                            "grip_idx": GRIP_TO_IDX[grip], "force_idx": FORCE_TO_IDX[force]})
            grip_counts[grip] = grip_counts.get(grip, 0) + 1
            force_counts[force] = force_counts.get(force, 0) + 1
    return {"records": records, "grip_counts": grip_counts, "force_counts": force_counts,
            "grip_weights": _weights(grip_counts, GRIPS),
            "force_weights": _weights(force_counts, FORCES), "skipped": skipped}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    idx = build_index(args.data_root, args.labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(idx), encoding="utf-8")
    print(f"index: {len(idx['records'])} records, {idx['skipped']} skipped, "
          f"grips={idx['grip_counts']}, forces={idx['force_counts']}")


if __name__ == "__main__":
    main()
