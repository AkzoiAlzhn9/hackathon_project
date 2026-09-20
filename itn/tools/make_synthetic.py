# -*- coding: utf-8 -*-
"""Generate synthetic data in the contest's CSV schema.

This is *not* a substitute for train.csv -- it exists so the pipeline can be
exercised and the metric verified without the real files (smoke tests, CI).
"""

import argparse
import csv
import os
import random

RU_NUM_SMALL = "один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать".split()
RU_NUM_TENS = "двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто".split()
KK_NUM_SMALL = "бір екі үш төрт бес алты жеті сегіз тоғыз он".split()
KK_NUM_TENS = "жиырма отыз қырық елу алпыс жетпіс сексен тоқсан".split()
RU_ORD = "первого второго третьего пятого десятого двадцатого".split()
KK_ORD = "бірінші екінші үшінші бесінші оныншы жиырмасыншы".split()
UNITS_RU = "метров литров километров градусов процентов тенге долларов минут".split()
UNITS_KK = "метр литр шақырым пайыз теңге келілік минут".split()
MONTHS_RU = "января февраля марта апреля мая июня июля".split()
MONTHS_KK = "қаңтар ақпан наурыз сәуір мамыр маусым".split()
FILLER_RU = "и в на он она мы они сказал говорил пришел видел дом работа очень просто потом сегодня завтра там тут около через для что как это был".split()
FILLER_KK = "және бұл ол біз олар айтты келді көрінеді деп үшін кейін бүгін ертең жұмыс үй өте жақсы сонымен қатар бір".split()
NAMES = "айгуль ержан марат динара алия нурлан асель тимур".split()
WHITELIST = ["кока кола", "ллама три", "си эм си", "телеграм", "смс", "аттеншн", "джипити", "опен эй ай"]
MAIL_DOMAIN = ["гмайл", "яндекс", "мэйл"]
MAIL_TLD = ["кз", "ру", "ком"]


def num_phrase(rng, kk):
    tens, small = (KK_NUM_TENS, KK_NUM_SMALL) if kk else (RU_NUM_TENS, RU_NUM_SMALL)
    roll = rng.random()
    if roll < 0.4:
        return [rng.choice(small)]
    if roll < 0.7:
        return [rng.choice(tens), rng.choice(small)]
    if roll < 0.9:
        return [rng.choice(small), "жүз" if kk else "сто", rng.choice(tens)]
    return [rng.choice(small), "мың" if kk else "тысяч", rng.choice(tens), rng.choice(small)]


def filler(rng, kk, n):
    pool = FILLER_KK if kk else FILLER_RU
    return [rng.choice(pool) for _ in range(n)]


def make_span(rng, kk):
    """Return (tokens, class) for one semiotic span, with its carrier words."""
    cls = rng.choices(
        ["CARDINAL", "ORDINAL", "DECIMAL", "DATE", "TIME", "MEASURE", "EMAIL", "WHITELIST"],
        weights=[26, 8, 5, 9, 9, 22, 6, 15],
    )[0]
    if cls == "CARDINAL":
        span = num_phrase(rng, kk)
        after = ["рет"] if kk else [rng.choice(["раза", "раз", "человека"])]
        return span, cls, [], after
    if cls == "ORDINAL":
        span = [rng.choice(KK_ORD if kk else RU_ORD)]
        return span, cls, [], ["орын"] if kk else ["место"]
    if cls == "DECIMAL":
        if kk:
            span = [rng.choice(KK_NUM_SMALL), "бүтін", rng.choice(KK_NUM_SMALL), "ондық"]
        else:
            span = [rng.choice(RU_NUM_SMALL), "целых", rng.choice(RU_NUM_SMALL), "десятых"]
        return span, cls, [], []
    if cls == "DATE":
        if kk:
            span = [rng.choice(KK_NUM_TENS), rng.choice(KK_NUM_SMALL), rng.choice(MONTHS_KK)]
        else:
            span = [rng.choice(RU_NUM_TENS), rng.choice(RU_ORD), rng.choice(MONTHS_RU)]
        return span, cls, [], []
    if cls == "TIME":
        span = num_phrase(rng, kk)[:1] + [rng.choice(KK_NUM_TENS if kk else RU_NUM_TENS)]
        before = ["сағат"] if kk else ["в"]
        return span, cls, before, []
    if cls == "MEASURE":
        span = num_phrase(rng, kk) + [rng.choice(UNITS_KK if kk else UNITS_RU)]
        return span, cls, [], []
    if cls == "EMAIL":
        span = [rng.choice(NAMES), "собачка", rng.choice(MAIL_DOMAIN), "точка", rng.choice(MAIL_TLD)]
        return span, cls, ["пошта"] if kk else ["почта"], []
    span = rng.choice(WHITELIST).split()
    return span, cls, [], []


def make_sentence(rng):
    kk = rng.random() < 0.45
    mixed = rng.random() < 0.15
    tokens, labels = [], []
    tokens += filler(rng, kk, rng.randint(0, 4))
    labels += ["O"] * len(tokens)

    for _ in range(rng.randint(1, 2) if rng.random() < 0.85 else 0):
        span, cls, before, after = make_span(rng, kk if not mixed else rng.random() < 0.5)
        tokens += before
        labels += ["O"] * len(before)
        tokens += span
        labels += [f"B-{cls}"] + [f"I-{cls}"] * (len(span) - 1)
        tokens += after
        labels += ["O"] * len(after)
        gap = filler(rng, kk, rng.randint(1, 4))
        tokens += gap
        labels += ["O"] * len(gap)

    if not tokens:
        tokens = filler(rng, kk, 4)
        labels = ["O"] * 4
    return tokens, labels


def write(path, sentences, with_labels, prefix):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["sent_id", "token_id", "token"] + (["label"] if with_labels else []))
        for i, (tokens, labels) in enumerate(sentences):
            sid = f"{prefix}{i:06d}"
            for j, token in enumerate(tokens):
                writer.writerow([sid, j, token] + ([labels[j]] if with_labels else []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/synthetic")
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--test", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    train = [make_sentence(rng) for _ in range(args.train)]
    test = [make_sentence(rng) for _ in range(args.test)]

    write(os.path.join(args.out_dir, "train.csv"), train, True, "t")
    write(os.path.join(args.out_dir, "test.csv"), test, False, "s")
    write(os.path.join(args.out_dir, "test_gold.csv"), test, True, "s")
    with open(os.path.join(args.out_dir, "sample_submission.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["sent_id", "token_id", "label"])
        for i, (tokens, _) in enumerate(test):
            for j in range(len(tokens)):
                writer.writerow([f"s{i:06d}", j, "O"])
    print(f"wrote synthetic data to {args.out_dir}")


if __name__ == "__main__":
    main()
