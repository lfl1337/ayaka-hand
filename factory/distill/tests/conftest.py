import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def crops_root(tmp_path: Path) -> Path:
    """Mini ayaka-crops layout: 6 crops over 3 classes, one broken row, one bad image."""
    rows = [
        ("crops/banana/1_1.jpg", "banana", "power", "firm"),
        ("crops/banana/1_2.jpg", "banana", "power", "delicate"),
        ("crops/apple/2_1.jpg", "apple", "tripod", "delicate"),
        ("crops/apple/2_2.jpg", "apple", "tripod", "delicate"),
        ("crops/knife/3_1.jpg", "knife", "lateral", "firm"),
        ("crops/knife/3_2.jpg", "knife", "no_grasp", "firm"),
    ]
    labels = []
    for rel, cls, grip, force in rows:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (120, 180, 60)).save(p, "JPEG")
        labels.append({"crop": rel, "class": cls,
                       "teacher": {"grip": grip, "force": force, "object_label": cls,
                                   "contact_region": "body", "contact_point": {"x": 0.5, "y": 0.5},
                                   "hazards": [], "rationale": "fixture"},
                       "prompt_version": "1.1"})
    # broken rows the builder must SKIP (count them): invalid grip enum, missing file
    labels.append({"crop": "crops/banana/1_1.jpg", "class": "banana",
                   "teacher": {"grip": "hook", "force": "firm"}})
    labels.append({"crop": "crops/ghost/9_9.jpg", "class": "ghost",
                   "teacher": {"grip": "power", "force": "firm"}})
    (tmp_path / "labels-test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in labels), encoding="utf-8")
    return tmp_path
