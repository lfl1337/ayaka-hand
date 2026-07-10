"""Accuracy + macro-F1 without sklearn (keeps the training env lean)."""


def accuracy(pred: list[int], true: list[int]) -> float:
    assert len(pred) == len(true) and pred
    return sum(p == t for p, t in zip(pred, true)) / len(true)


def per_class_f1(pred: list[int], true: list[int], n_classes: int) -> list[float]:
    out = []
    for c in range(n_classes):
        tp = sum(p == c and t == c for p, t in zip(pred, true))
        fp = sum(p == c and t != c for p, t in zip(pred, true))
        fn = sum(p != c and t == c for p, t in zip(pred, true))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return out


def macro_f1(pred: list[int], true: list[int], n_classes: int) -> float:
    f1 = per_class_f1(pred, true, n_classes)
    seen = [c for c in range(n_classes) if c in true or c in pred]
    return sum(f1[c] for c in seen) / len(seen) if seen else 0.0
