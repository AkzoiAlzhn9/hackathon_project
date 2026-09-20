"""Reading and writing the contest CSV files."""

import csv
import gzip
import sys
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


class Sentence(NamedTuple):
    sent_id: str
    token_ids: List[str]
    tokens: List[str]
    labels: Optional[List[str]]


def _open(path: str):
    """Open a CSV, transparently handling gzip (the contest files compress ~4x)."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def _rows(path: str) -> Iterator[Dict[str, str]]:
    with _open(path) as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return
        missing = {"sent_id", "token_id", "token"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        for row in reader:
            yield row


def read_sentences(path: str, limit: Optional[int] = None) -> List[Sentence]:
    """Group a token-per-row CSV into sentences, preserving file order of sent_ids.

    Tokens are ordered by numeric token_id when it parses as an int, otherwise by
    the order they appear in the file.
    """
    order: List[str] = []
    buckets: Dict[str, List[Dict[str, str]]] = {}
    has_label = False

    for row in _rows(path):
        sid = row["sent_id"]
        if sid not in buckets:
            if limit is not None and len(order) >= limit:
                break
            buckets[sid] = []
            order.append(sid)
        if row.get("label"):
            has_label = True
        buckets[sid].append(row)

    sentences: List[Sentence] = []
    for sid in order:
        rows = buckets[sid]
        try:
            rows.sort(key=lambda r: int(r["token_id"]))
        except (TypeError, ValueError):
            pass
        tokens = [(r["token"] or "") for r in rows]
        token_ids = [r["token_id"] for r in rows]
        labels = [(r.get("label") or "O") for r in rows] if has_label else None
        sentences.append(Sentence(sid, token_ids, tokens, labels))
    return sentences


def write_submission(path: str, sentences: Sequence[Sentence], predictions: Sequence[Sequence[str]]) -> int:
    """Write solution.csv: exactly one row per input token."""
    if len(sentences) != len(predictions):
        raise ValueError("sentence/prediction count mismatch")
    written = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["sent_id", "token_id", "label"])
        for sent, preds in zip(sentences, predictions):
            if len(sent.tokens) != len(preds):
                raise ValueError(f"{sent.sent_id}: {len(sent.tokens)} tokens vs {len(preds)} labels")
            for token_id, label in zip(sent.token_ids, preds):
                writer.writerow([sent.sent_id, token_id, label])
                written += 1
    return written


def split_dev(sentences: List[Sentence], dev_size: int, seed: int = 13):
    """Deterministic train/dev split on whole sentences."""
    import random

    if dev_size <= 0 or dev_size >= len(sentences):
        return sentences, []
    rng = random.Random(seed)
    idx = list(range(len(sentences)))
    rng.shuffle(idx)
    dev_idx = set(idx[:dev_size])
    train = [s for i, s in enumerate(sentences) if i not in dev_idx]
    dev = [s for i, s in enumerate(sentences) if i in dev_idx]
    return train, dev
