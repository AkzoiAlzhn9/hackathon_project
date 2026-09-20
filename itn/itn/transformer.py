# -*- coding: utf-8 -*-
"""Optional transformer tagger (XLM-R / other multilingual encoder).

Higher ceiling than the CRF, needs a GPU to be practical.  Usage:

    pip install torch transformers
    python -m itn.transformer train --train data/train.csv --out-dir runs/xlmr
    python -m itn.transformer predict --model-dir runs/xlmr --test data/test.csv --out solution.csv

NOTE: this module could not be executed in the environment it was written in
(no GPU and huggingface.co unreachable), unlike itn/crf.py which is covered by
`python -m itn.cli selfcheck`.  Treat the first run as a smoke test: train on
`--max-sents 2000` first and confirm the dev report looks sane.
"""

import argparse
import json
import os
from typing import List, Sequence

from . import LABELS
from .data import Sentence, read_sentences, split_dev, write_submission
from .spans import format_report, repair, score

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
MAX_WORDS = 128  # sentences are short; longer ones are tagged in consecutive chunks


def _chunks(tokens: Sequence[str], size: int = MAX_WORDS) -> List[List[str]]:
    return [list(tokens[i : i + size]) for i in range(0, len(tokens), size)] or [[]]


def _encode(tokenizer, words: Sequence[str], labels=None, max_length: int = 256):
    encoding = tokenizer(
        list(words),
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    word_ids = encoding.word_ids()
    if labels is not None:
        aligned, previous = [], None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != previous:
                aligned.append(LABEL2ID.get(labels[word_id], 0))
            else:
                aligned.append(-100)  # score only the first sub-token of a word
            previous = word_id
        encoding["labels"] = aligned
    return encoding, word_ids


def _collate(batch, pad_id: int):
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    input_ids, attention, labels = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        attention.append(item["attention_mask"] + [0] * pad)
        if "labels" in item:
            labels.append(item["labels"] + [-100] * pad)
    out = {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention),
    }
    if labels:
        out["labels"] = torch.tensor(labels)
    return out


def train(args) -> int:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

    sentences = read_sentences(args.train, limit=args.max_sents)
    train_set, dev_set = split_dev(sentences, args.dev_size, seed=args.seed)
    print(f"{len(train_set)} train / {len(dev_set)} dev sentences", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    examples = []
    for sent in train_set:
        for chunk_start in range(0, max(len(sent.tokens), 1), MAX_WORDS):
            words = sent.tokens[chunk_start : chunk_start + MAX_WORDS]
            labels = sent.labels[chunk_start : chunk_start + MAX_WORDS]
            if not words:
                continue
            encoding, _ = _encode(tokenizer, words, labels)
            examples.append(dict(encoding))

    pad_id = tokenizer.pad_token_id or 0
    loader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, pad_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "itn_labels.json"), "w", encoding="utf-8") as fh:
        json.dump(list(LABELS), fh, ensure_ascii=False)
    print(f"saved to {args.out_dir}", flush=True)

    if dev_set:
        predictions = _predict(model, tokenizer, device, dev_set, args.batch_size)
        print("\n--- dev ---")
        print(format_report(score([s.labels for s in dev_set], predictions)))
    return 0


def _predict(model, tokenizer, device, sentences: Sequence[Sentence], batch_size: int) -> List[List[str]]:
    import torch

    model.eval()
    results: List[List[str]] = []
    # Flatten to (sentence index, word chunk) so batching stays dense.
    flat = []
    for index, sent in enumerate(sentences):
        for chunk in _chunks(sent.tokens):
            flat.append((index, chunk))
        results.append([])

    pad_id = tokenizer.pad_token_id or 0
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            window = flat[start : start + batch_size]
            encodings, word_id_lists = [], []
            for _, words in window:
                encoding, word_ids = _encode(tokenizer, words or ["."])
                encodings.append(dict(encoding))
                word_id_lists.append(word_ids)
            batch = _collate(encodings, pad_id)
            batch = {key: value.to(device) for key, value in batch.items()}
            predicted = model(**batch).logits.argmax(-1).cpu().tolist()

            for (index, words), word_ids, row in zip(window, word_id_lists, predicted):
                labels = ["O"] * len(words)
                previous = None
                for position, word_id in enumerate(word_ids):
                    if word_id is not None and word_id != previous and word_id < len(labels):
                        labels[word_id] = ID2LABEL.get(row[position], "O")
                    previous = word_id
                results[index].extend(labels)

    return [repair(labels) for labels in results]


def predict(args) -> int:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    sentences = read_sentences(args.test)
    predictions = _predict(model, tokenizer, device, sentences, args.batch_size)
    rows = write_submission(args.out, sentences, predictions)
    print(f"wrote {args.out}: {rows} rows", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m itn.transformer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--train", required=True)
    p_train.add_argument("--out-dir", default="runs/xlmr")
    p_train.add_argument("--model-name", default="xlm-roberta-base")
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=3e-5)
    p_train.add_argument("--max-sents", type=int, default=None)
    p_train.add_argument("--dev-size", type=int, default=5000)
    p_train.add_argument("--seed", type=int, default=13)
    p_train.add_argument("--device", default=None)
    p_train.add_argument("--log-every", type=int, default=200)
    p_train.set_defaults(func=train)

    p_predict = sub.add_parser("predict")
    p_predict.add_argument("--model-dir", required=True)
    p_predict.add_argument("--test", required=True)
    p_predict.add_argument("--out", default="solution.csv")
    p_predict.add_argument("--batch-size", type=int, default=64)
    p_predict.add_argument("--device", default=None)
    p_predict.set_defaults(func=predict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
