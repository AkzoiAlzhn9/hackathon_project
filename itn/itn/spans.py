"""BIO <-> span conversion, BIO repair and the contest's span-level F1 metric."""

from typing import Dict, List, Sequence, Set, Tuple

Span = Tuple[int, int, str]  # (start, end_inclusive, class)


def _parse(label: str) -> Tuple[str, str]:
    if not label or label == "O":
        return "O", ""
    if len(label) > 2 and label[1] == "-" and label[0] in ("B", "I"):
        return label[0], label[2:]
    return "O", ""


def to_spans(labels: Sequence[str]) -> List[Span]:
    """Decode BIO exactly as the judge does: an I- that does not continue a span
    of the same class starts a new span."""
    spans: List[Span] = []
    start = -1
    cls = ""
    for i, label in enumerate(labels):
        prefix, this_cls = _parse(label)
        if prefix == "O":
            if start >= 0:
                spans.append((start, i - 1, cls))
                start, cls = -1, ""
            continue
        if prefix == "B" or start < 0 or this_cls != cls:
            if start >= 0:
                spans.append((start, i - 1, cls))
            start, cls = i, this_cls
    if start >= 0:
        spans.append((start, len(labels) - 1, cls))
    return spans


def from_spans(spans: Sequence[Span], length: int) -> List[str]:
    labels = ["O"] * length
    for start, end, cls in spans:
        labels[start] = f"B-{cls}"
        for i in range(start + 1, end + 1):
            labels[i] = f"I-{cls}"
    return labels


def repair(labels: Sequence[str]) -> List[str]:
    """Canonicalise a label sequence (I- at a span start becomes B-).

    Scoring-neutral by the problem statement, but it keeps outputs clean and makes
    predicted spans easy to inspect.
    """
    return from_spans(to_spans(labels), len(labels))


def score(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Span-level micro F1 with exact boundaries, plus a per-class breakdown."""
    tp = fp = fn = 0
    per_class: Dict[str, List[int]] = {}

    for gold_labels, pred_labels in zip(gold, pred):
        gold_spans: Set[Span] = set(to_spans(gold_labels))
        pred_spans: Set[Span] = set(to_spans(pred_labels))
        for span in gold_spans & pred_spans:
            per_class.setdefault(span[2], [0, 0, 0])[0] += 1
        for span in pred_spans - gold_spans:
            per_class.setdefault(span[2], [0, 0, 0])[1] += 1
        for span in gold_spans - pred_spans:
            per_class.setdefault(span[2], [0, 0, 0])[2] += 1
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)

    def prf(t: int, p: int, n: int) -> Tuple[float, float, float]:
        if t == 0:
            return 0.0, 0.0, 0.0
        precision = t / (t + p)
        recall = t / (t + n)
        return precision, recall, 2 * precision * recall / (precision + recall)

    precision, recall, f1 = prf(tp, fp, fn)
    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "score": round(100 * f1, 2),
        "per_class": {
            cls: dict(
                zip(
                    ("precision", "recall", "f1"),
                    prf(*counts),
                ),
                support=counts[0] + counts[2],
            )
            for cls, counts in sorted(per_class.items())
        },
    }
    return result


def format_report(result: Dict[str, float]) -> str:
    lines = [
        f"span F1 = {result['f1']:.4f}   (contest score {result['score']:.2f})",
        f"precision = {result['precision']:.4f}  recall = {result['recall']:.4f}",
        f"TP={result['tp']}  FP={result['fp']}  FN={result['fn']}",
        "",
        f"{'class':<12}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}",
    ]
    for cls, stats in result["per_class"].items():
        lines.append(
            f"{cls:<12}{stats['precision']:>8.3f}{stats['recall']:>8.3f}"
            f"{stats['f1']:>8.3f}{stats['support']:>9}"
        )
    return "\n".join(lines)
