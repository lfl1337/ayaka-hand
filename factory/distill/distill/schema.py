"""Canonical label spaces (Schema v1, FROZEN) + the lookup-table strawman baseline.

Index order is load-bearing: the student's head outputs, the ONNX metadata and
the eval tables all use these positions. Do not reorder.
"""

GRIPS = ["power", "lateral", "pinch", "tripod", "no_grasp"]
FORCES = ["delicate", "firm"]

GRIP_TO_IDX = {g: i for i, g in enumerate(GRIPS)}
FORCE_TO_IDX = {f: i for i, f in enumerate(FORCES)}

# Port of web/src/perception/lookup.ts MAP (2026-07-09) — the Tag-4 ablation strawman.
# Baseline gets the GROUND-TRUTH class name (perfect-detector assumption = strongest
# possible strawman); unknown class -> safe refusal, exactly like lookupGrip().
LOOKUP_BASELINE: dict[str, tuple[str, str]] = {
    "cup": ("power", "firm"),
    "bottle": ("power", "firm"),
    "wine glass": ("power", "delicate"),
    "bowl": ("power", "firm"),
    "banana": ("power", "delicate"),
    "apple": ("tripod", "firm"),
    "orange": ("tripod", "firm"),
    "donut": ("tripod", "delicate"),
    "cell phone": ("lateral", "firm"),
    "book": ("lateral", "firm"),
    "remote": ("power", "firm"),
    "scissors": ("lateral", "firm"),
    "knife": ("lateral", "firm"),
    "fork": ("lateral", "firm"),
    "spoon": ("lateral", "firm"),
    "toothbrush": ("lateral", "firm"),
    "mouse": ("power", "firm"),
    "keyboard": ("no_grasp", "firm"),
    "laptop": ("no_grasp", "firm"),
    "sports ball": ("power", "firm"),
    "frisbee": ("pinch", "firm"),
    "teddy bear": ("power", "delicate"),
    "vase": ("power", "delicate"),
    "carrot": ("tripod", "firm"),
    "mug": ("power", "firm"),
    "drinking glass": ("power", "delicate"),
    "pen": ("lateral", "firm"),
    "key": ("pinch", "firm"),
    "headphone case": ("power", "firm"),
    "egg": ("tripod", "delicate"),
    "plate": ("lateral", "delicate"),
    "jar": ("power", "firm"),
}

def lookup_baseline(cls: str) -> tuple[str, str]:
    return LOOKUP_BASELINE.get(cls, ("no_grasp", "firm"))
