# -*- coding: utf-8 -*-
"""CRF sequence tagger (python-crfsuite backend)."""

import time
from typing import Iterable, List, Optional, Sequence

from .data import Sentence
from .features import sentence_features
from .spans import repair

DEFAULT_PARAMS = {
    "c1": 0.1,
    "c2": 0.05,
    "max_iterations": 150,
    "feature.minfreq": 2,
    "feature.possible_transitions": True,
    "feature.possible_states": False,
}


def _import_crfsuite():
    try:
        import pycrfsuite  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "python-crfsuite is required.  Install it with:\n"
            "    pip install python-crfsuite\n"
            f"(import failed: {exc})"
        )
    return pycrfsuite


def train(
    sentences: Iterable[Sentence],
    model_path: str,
    algorithm: str = "lbfgs",
    params: Optional[dict] = None,
    verbose: bool = True,
) -> str:
    pycrfsuite = _import_crfsuite()

    trainer = pycrfsuite.Trainer(verbose=verbose)
    trainer.select(algorithm, "crf1d")
    settings = dict(DEFAULT_PARAMS)
    if algorithm != "lbfgs":
        # c1/max_iterations are lbfgs-only knobs; crfsuite rejects unknown ones.
        settings = {
            key: value
            for key, value in settings.items()
            if key not in ("c1", "c2", "max_iterations")
        }
    if params:
        settings.update(params)
    for key, value in settings.items():
        try:
            trainer.set(key, str(value).lower() if isinstance(value, bool) else str(value))
        except Exception:  # pragma: no cover - unsupported knob for this algorithm
            pass

    started = time.time()
    count = 0
    for sent in sentences:
        if sent.labels is None:
            raise ValueError(f"sentence {sent.sent_id} has no labels")
        # Append one sentence at a time so Python-side feature lists are freed
        # immediately; crfsuite keeps its own compact copy.
        trainer.append(sentence_features(sent.tokens), list(sent.labels))
        count += 1
        if verbose and count % 50000 == 0:
            print(f"  featurised {count} sentences ({time.time() - started:.0f}s)", flush=True)

    if count == 0:
        raise ValueError("no training sentences")
    trainer.train(model_path)
    return model_path


class Tagger:
    def __init__(self, model_path: str):
        pycrfsuite = _import_crfsuite()
        self._tagger = pycrfsuite.Tagger()
        self._tagger.open(model_path)

    def predict_tokens(self, tokens: Sequence[str]) -> List[str]:
        if not tokens:
            return []
        return repair(self._tagger.tag(sentence_features(tokens)))

    def predict(self, sentences: Sequence[Sentence], verbose: bool = False) -> List[List[str]]:
        out: List[List[str]] = []
        started = time.time()
        for i, sent in enumerate(sentences, 1):
            out.append(self.predict_tokens(sent.tokens))
            if verbose and i % 50000 == 0:
                print(f"  tagged {i} sentences ({time.time() - started:.0f}s)", flush=True)
        return out

    def close(self) -> None:
        self._tagger.close()
