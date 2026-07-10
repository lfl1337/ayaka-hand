#!/usr/bin/env bash
# web/scripts/fetch-student.sh — place the distilled student (kd-v3) into web/public/.
#
# The student ONNX is already BUNDLED in this repository at
#   web/public/models/student/student.onnx
# so a fresh clone runs out of the box and you normally do not need this script.
#
# To REGENERATE it from scratch, run the distillation pipeline in factory/distill
# (train on the teacher labels, then export), and copy the result here:
#
#   cd ../factory/distill
#   uv sync
#   uv run python -m distill.train       --data-root <crops> --train-labels <train.jsonl> \
#                                        --val-labels <val.jsonl> --out-dir runs/kd-v3 \
#                                        --arch timm_mnv2_100 --epochs 80
#   uv run python -m distill.export_onnx --checkpoint runs/kd-v3/best.pt \
#                                        --out runs/kd-v3/student.onnx
#   cp runs/kd-v3/student.onnx ../../web/public/models/student/student.onnx
#
# See factory/distill and docs/benchmarks/distill-before-after.md for details.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p public/models/student
if [ -f public/models/student/student.onnx ]; then
  echo "student.onnx already present:"
  ls -lh public/models/student/student.onnx
else
  echo "student.onnx is missing — regenerate it via factory/distill (see the comment in this script)." >&2
  exit 1
fi
