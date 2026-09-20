# -*- coding: utf-8 -*-
"""Command line entry point:  python -m itn.cli <command> [options]"""

import argparse
import csv
import os
import sys
import tempfile
import time
from typing import List

from . import crf
from .data import read_sentences, split_dev, write_submission
from .spans import format_report, score


def _log(message: str) -> None:
    print(message, flush=True)


# --- commands -------------------------------------------------------------


def cmd_train(args) -> int:
    started = time.time()
    _log(f"reading {args.train} ...")
    sentences = read_sentences(args.train, limit=args.max_sents)
    _log(f"  {len(sentences)} sentences, {sum(len(s.tokens) for s in sentences)} tokens")

    train_set, dev_set = split_dev(sentences, args.dev_size, seed=args.seed)
    if dev_set:
        _log(f"  holding out {len(dev_set)} sentences for dev")

    params = {}
    if args.c1 is not None:
        params["c1"] = args.c1
    if args.c2 is not None:
        params["c2"] = args.c2
    if args.max_iterations is not None:
        params["max_iterations"] = args.max_iterations
    if args.min_freq is not None:
        params["feature.minfreq"] = args.min_freq

    _log(f"training CRF ({args.algorithm}) -> {args.model}")
    crf.train(train_set, args.model, algorithm=args.algorithm, params=params, verbose=not args.quiet)
    _log(f"model written: {args.model} ({os.path.getsize(args.model) / 1e6:.1f} MB)")

    if dev_set:
        tagger = crf.Tagger(args.model)
        predictions = tagger.predict(dev_set)
        result = score([s.labels for s in dev_set], predictions)
        _log("\n--- dev ---")
        _log(format_report(result))
    _log(f"\ndone in {time.time() - started:.0f}s")
    return 0


def cmd_predict(args) -> int:
    started = time.time()
    _log(f"reading {args.test} ...")
    sentences = read_sentences(args.test)
    _log(f"  {len(sentences)} sentences, {sum(len(s.tokens) for s in sentences)} tokens")

    tagger = crf.Tagger(args.model)
    predictions = tagger.predict(sentences, verbose=not args.quiet)
    rows = write_submission(args.out, sentences, predictions)
    spans = sum(len(set(_spans(p))) for p in predictions)
    _log(f"wrote {args.out}: {rows} rows, {spans} predicted spans ({time.time() - started:.0f}s)")
    if spans == 0:
        _log("WARNING: no spans predicted -- an all-O submission scores 0.00")
    return 0


def _spans(labels):
    from .spans import to_spans

    return to_spans(labels)


def cmd_evaluate(args) -> int:
    gold_sentences = read_sentences(args.gold)
    if gold_sentences and gold_sentences[0].labels is None:
        _log(f"{args.gold} has no 'label' column")
        return 2

    pred_labels = {}
    with open(args.pred, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pred_labels.setdefault(row["sent_id"], {})[str(row["token_id"])] = row["label"]

    gold_seqs: List[List[str]] = []
    pred_seqs: List[List[str]] = []
    missing = 0
    for sent in gold_sentences:
        per_sent = pred_labels.get(sent.sent_id, {})
        seq = []
        for token_id in sent.token_ids:
            label = per_sent.get(str(token_id))
            if label is None:
                missing += 1
                label = "O"
            seq.append(label)
        gold_seqs.append(list(sent.labels))
        pred_seqs.append(seq)

    if missing:
        _log(f"WARNING: {missing} tokens missing from {args.pred} (scored as O)")
    _log(format_report(score(gold_seqs, pred_seqs)))
    return 0


def cmd_validate(args) -> int:
    """Check a submission against test.csv the way the judge does."""
    expected = set()
    with open(args.test, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            expected.add((row["sent_id"], str(row["token_id"])))

    from . import LABELS

    allowed = set(LABELS)
    seen = set()
    duplicates = 0
    bad_labels = set()
    with open(args.pred, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["sent_id", "token_id", "label"]:
            _log(f"FAIL: header is {reader.fieldnames}, expected ['sent_id', 'token_id', 'label']")
            return 1
        for row in reader:
            key = (row["sent_id"], str(row["token_id"]))
            if key in seen:
                duplicates += 1
            seen.add(key)
            if row["label"] not in allowed:
                bad_labels.add(row["label"])

    problems = []
    if duplicates:
        problems.append(f"{duplicates} duplicate rows")
    if bad_labels:
        problems.append(f"invalid labels: {sorted(bad_labels)[:5]}")
    if seen - expected:
        problems.append(f"{len(seen - expected)} rows not present in {args.test}")
    if expected - seen:
        problems.append(f"{len(expected - seen)} tokens of {args.test} have no row")

    if problems:
        for problem in problems:
            _log(f"FAIL: {problem}")
        return 1
    _log(f"OK: {len(seen)} rows, one per test token, all labels valid")
    return 0


def cmd_run(args) -> int:
    """train + predict + validate in one call."""
    model = args.model or os.path.join(tempfile.gettempdir(), "itn-model.crf")
    train_args = argparse.Namespace(
        train=args.train,
        model=model,
        max_sents=args.max_sents,
        dev_size=args.dev_size,
        seed=args.seed,
        algorithm=args.algorithm,
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iterations,
        min_freq=args.min_freq,
        quiet=args.quiet,
    )
    rc = cmd_train(train_args)
    if rc:
        return rc
    rc = cmd_predict(argparse.Namespace(model=model, test=args.test, out=args.out, quiet=args.quiet))
    if rc:
        return rc
    return cmd_validate(argparse.Namespace(test=args.test, pred=args.out))


def cmd_selfcheck(args) -> int:
    """End-to-end check on generated data -- no contest files needed."""
    import subprocess

    tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    workdir = args.work_dir or tempfile.mkdtemp(prefix="itn-selfcheck-")
    _log(f"work dir: {workdir}")
    subprocess.check_call(
        [sys.executable, os.path.join(tools, "make_synthetic.py"), "--out-dir", workdir,
         "--train", str(args.train_size), "--test", str(args.test_size)]
    )

    model = os.path.join(workdir, "model.crf")
    solution = os.path.join(workdir, "solution.csv")
    rc = cmd_run(
        argparse.Namespace(
            train=os.path.join(workdir, "train.csv"),
            test=os.path.join(workdir, "test.csv"),
            out=solution,
            model=model,
            max_sents=None,
            dev_size=0,
            seed=13,
            algorithm="lbfgs",
            c1=None,
            c2=None,
            max_iterations=80,
            min_freq=1,
            quiet=True,
        )
    )
    if rc:
        return rc
    _log("\n--- held-out synthetic test ---")
    cmd_evaluate(argparse.Namespace(gold=os.path.join(workdir, "test_gold.csv"), pred=solution))
    return 0


# --- argument parsing -----------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m itn.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_training_options(p):
        p.add_argument("--algorithm", default="lbfgs", choices=["lbfgs", "l2sgd", "ap", "pa", "arow"])
        p.add_argument("--max-sents", type=int, default=None, help="train on the first N sentences only")
        p.add_argument("--dev-size", type=int, default=5000, help="sentences held out for dev scoring")
        p.add_argument("--seed", type=int, default=13)
        p.add_argument("--c1", type=float, default=None)
        p.add_argument("--c2", type=float, default=None)
        p.add_argument("--max-iterations", type=int, default=None)
        p.add_argument("--min-freq", type=int, default=None, help="drop features seen fewer than N times")
        p.add_argument("--quiet", action="store_true")

    p_train = sub.add_parser("train", help="train the CRF tagger")
    p_train.add_argument("--train", required=True)
    p_train.add_argument("--model", default="model.crf")
    add_training_options(p_train)
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="tag test.csv and write solution.csv")
    p_predict.add_argument("--model", default="model.crf")
    p_predict.add_argument("--test", required=True)
    p_predict.add_argument("--out", default="solution.csv")
    p_predict.add_argument("--quiet", action="store_true")
    p_predict.set_defaults(func=cmd_predict)

    p_run = sub.add_parser("run", help="train + predict + validate")
    p_run.add_argument("--train", required=True)
    p_run.add_argument("--test", required=True)
    p_run.add_argument("--out", default="solution.csv")
    p_run.add_argument("--model", default=None)
    add_training_options(p_run)
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("evaluate", help="span F1 of a submission against a labelled file")
    p_eval.add_argument("--gold", required=True, help="CSV with a label column")
    p_eval.add_argument("--pred", required=True, help="submission CSV")
    p_eval.set_defaults(func=cmd_evaluate)

    p_validate = sub.add_parser("validate", help="check submission shape against test.csv")
    p_validate.add_argument("--test", required=True)
    p_validate.add_argument("--pred", required=True)
    p_validate.set_defaults(func=cmd_validate)

    p_self = sub.add_parser("selfcheck", help="end-to-end run on generated data")
    p_self.add_argument("--work-dir", default=None)
    p_self.add_argument("--train-size", type=int, default=4000)
    p_self.add_argument("--test-size", type=int, default=800)
    p_self.set_defaults(func=cmd_selfcheck)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
