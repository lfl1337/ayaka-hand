import json

from distill.build_dataset import build_index


def test_build_index_records_and_weights(crops_root):
    idx = build_index(crops_root, "labels-test.jsonl")
    assert len(idx["records"]) == 6
    assert idx["skipped"] == 2                       # bad enum + missing file
    r0 = idx["records"][0]
    assert set(r0) == {"path", "class", "grip_idx", "force_idx"}
    assert (crops_root / r0["path"]).exists()
    assert idx["grip_counts"] == {"power": 2, "tripod": 2, "lateral": 1, "no_grasp": 1}
    # inverse-frequency weights, mean-normalized, 0 count -> weight 0 (never predicted, never penalized)
    assert len(idx["grip_weights"]) == 5 and idx["grip_weights"][2] == 0.0  # pinch unseen
    assert len(idx["force_weights"]) == 2
    assert abs(sum(w for w in idx["grip_weights"] if w) / 4 - 1.0) < 1e-6


def test_malformed_json_line_is_skipped_and_counted(crops_root):
    # docstring promises invalid rows are skipped AND counted, not raised on
    src = (crops_root / "labels-test.jsonl").read_text(encoding="utf-8")
    corrupt = crops_root / "labels-corrupt.jsonl"
    corrupt.write_text(src + "\n{not json", encoding="utf-8")
    idx = build_index(crops_root, "labels-corrupt.jsonl")
    assert len(idx["records"]) == 6                  # unchanged: valid rows still parse
    assert idx["skipped"] == 3                       # bad enum + missing file + corrupt line


def test_cli_writes_json(crops_root, capsys):
    from distill.build_dataset import main
    out = crops_root / "index.json"
    main(["--data-root", str(crops_root), "--labels", "labels-test.jsonl", "--out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["records"]) == 6
