# ITN — BIO span tagging for Kazakh/Russian spoken text

Solution pipeline for the contest task *«Обратная нормализация текста»* (problem D):
given a sentence as a token sequence, label every token `O` / `B-<CLASS>` / `I-<CLASS>`
over the eight semiotic classes `CARDINAL ORDINAL DECIMAL DATE TIME MEASURE EMAIL
WHITELIST`. Scored by span-level F1 with exact boundaries.

## Status

The contest data (`train.csv`, `test.csv`) is **not** in this repository and
`official.contest.yandex.ru` is not reachable from the environment this was written
in, so no `solution.csv` for the real test set was produced here and no real-data F1
is claimed anywhere in this README. What *is* verified:

* `python -m unittest discover -s tests` — 21 tests, all passing. They pin the metric
  against the statement (exact boundaries only; an `I-` that does not continue a span
  of the same class starts a new one), the CSV contract, and the feature contract.
* `python -m itn.cli selfcheck` — generates data in the contest's schema, trains,
  predicts, validates the submission shape and scores it. Passes end to end.
  Its F1 is near 1.0 because the generator is template-based; it proves the
  **plumbing**, not accuracy on real speech transcripts.
* `itn/transformer.py` is the exception: it is **untested** — no GPU and
  `huggingface.co` was blocked. Smoke-test it with `--max-sents 2000` before a
  full run.

## Quickstart

```bash
pip install -r requirements.txt          # python-crfsuite; CPU only
# put the contest files in data/: train.csv, test.csv

python -m itn.cli run \
    --train data/train.csv \
    --test  data/test.csv \
    --out   solution.csv
```

`run` = train + predict + validate. It prints a dev report on a held-out slice of
`train.csv` (5000 sentences by default) — that number is the one to trust while
iterating, and it ends with a shape check of `solution.csv` against `test.csv`
(one row per token, no duplicates, no invalid labels — the three things the judge
rejects outright).

`sample_submission.csv` also works as the index to validate against, since it
carries the same `sent_id`/`token_id` pairs as `test.csv`:

```bash
python -m itn.cli validate --test data/sample_submission.csv --pred solution.csv
```

Separate steps, if you prefer:

```bash
python -m itn.cli train    --train data/train.csv --model model.crf
python -m itn.cli predict  --model model.crf --test data/test.csv --out solution.csv
python -m itn.cli validate --test data/test.csv --pred solution.csv
python -m itn.cli evaluate --gold some_labelled.csv --pred solution.csv
```

Submit `solution.csv` only — the statement asks for the file, not the code.

## Approach

A linear-chain CRF (`python-crfsuite`) over the 17 labels. The task is a
context-disambiguation problem more than a lexicon problem — the statement's own
examples put the same word `три` in four different classes, decided purely by its
neighbours — so the features are built around context:

| group | features |
|---|---|
| identity | word at offsets −2…+2 |
| context | bigrams `w[-1]\|w[0]`, `w[0]\|w[1]`, `w[-2]\|w[-1]`, `w[1]\|w[2]`, skip-gram `w[-1]\|w[+1]` |
| lexicon | per-offset tags: `NUM MULT ORD UNIT CUR TIMEW MONTH YEARW MAIL DEC`, plus tag bigrams/trigrams |
| morphology | prefixes/suffixes of length 2–4 (both languages inflect heavily; Kazakh case suffixes are stripped before lexicon lookup) |
| position | BOS/EOS, position from either end, sentence length |
| sentence | `sent_has_mail` + which side the email markers are on (dictated addresses put `собачка`/`точка` several tokens away), `sent_has_num`, numeral run length around the token |

Lexicons (`itn/lexicons.py`) are not the classifier — the CRF learns word identities
from 738k sentences by itself. They exist so *unseen inflected forms* still generalise:
`мыңға` reaches `мың`, `пятидесятого` is recognised as an ordinal.

Predictions are canonicalised through `spans.repair()` so an orphan `I-` is written
as `B-`. That is scoring-neutral by the statement, but it keeps the output inspectable.

Why a CRF first: the decision is local and contextual, transitions matter (exact span
boundaries are what is scored, and BIO legality is a transition constraint), it trains
on CPU in minutes, and it gives an honest dev number to iterate against without a GPU.

### Optional: transformer

`itn/transformer.py` fine-tunes a multilingual encoder (default `xlm-roberta-base`)
for token classification — labels on the first sub-token of each word, word-level
chunking at 128 words, greedy decode, same `repair()`. This has the higher ceiling on
mixed Kazakh/Russian and should beat the CRF given a GPU, but see **Status**: it has
not been executed.

```bash
pip install torch transformers
python -m itn.transformer train   --train data/train.csv --out-dir runs/xlmr --max-sents 2000  # smoke test
python -m itn.transformer train   --train data/train.csv --out-dir runs/xlmr
python -m itn.transformer predict --model-dir runs/xlmr --test data/test.csv --out solution.csv
```

## Scaling and tuning

Measured on this machine (4 cores, no GPU, synthetic data at ~9 tokens/sentence):

| workload | result |
|---|---|
| feature extraction | ~112k tokens/s |
| train, 150k sentences / 1.36M tokens, 100 L-BFGS iterations | 220 s wall, 1.23 GB peak RSS |
| tagging | 20k sentences in 3.2 s |
| validating a 1.88M-row submission | 10 s |

The real `test.csv` index is 146,797 sentences / 1,881,565 tokens (see **Dataset
shape**), so `train.csv` is on the order of 9.5M tokens: about 85 s to featurise, and
extrapolating the training row gives roughly 25 minutes at 100 iterations. Memory will
*not* scale as gently: peak RSS is driven by the number of distinct features, and real
transcripts have a far larger vocabulary than the generator's. If you hit swap, raise
`--min-freq` or cap `--max-sents` first. Knobs:

* `--max-sents N` — train on a prefix of the data (fast iteration, or a memory cap).
* `--min-freq N` — drop features seen fewer than N times. Raise it (3–5) if memory
  is tight or the model file gets large; it costs a little accuracy.
* `--max-iterations`, `--c1`, `--c2` — L-BFGS budget and L1/L2 regularisation.
  Defaults: 150 iterations, c1=0.1, c2=0.05.
* `--algorithm ap` (averaged perceptron) or `l2sgd` — much faster than `lbfgs`,
  usually a bit worse. Good for a first pass over the full data.
* `--dev-size N` — held-out sentences for the dev report; set `0` to train on everything
  once the configuration is settled.

Where to look for gains, in rough order of expected value: per-class errors in the dev
report (`evaluate` breaks F1 down by class — boundary errors concentrate in MEASURE and
DATE, where the unit or month word is in the span but the counted object is not);
then the transformer; then an ensemble of the two.

## Dataset shape

Measured from the real `sample_submission.csv` (the only contest file available here —
it carries the test index but no tokens, so it cannot be used to train or predict):

* 146,797 sentences, 1,881,565 tokens, mean 12.8 tokens/sentence, min 2, max 182.
* `token_id` runs 0…n−1 in every sentence, with no gaps.
* `sent_id` is **not** contiguous: the ids span `s000000`…`s164999` with 18,203 values
  unused. Do not derive a sentence count from the id range, and do not assume a
  `sent_id` is unique across `train.csv` and `test.csv` — nothing in the data says the
  two files share one id space.
* Sentences are longer than the generator's ~9 tokens, and the longest exceeds the
  transformer's 128-word chunk, so the chunking path in `itn/transformer.py` is live
  on real data rather than dead code.

## Layout

```
itn/
  itn/__init__.py       label set
  itn/data.py           CSV reading/grouping, submission writing, dev split
  itn/lexicons.py       numerals, units, currency, months, email markers; Kazakh stemming
  itn/features.py       CRF features (cached lexicon lookups)
  itn/crf.py            training and tagging
  itn/spans.py          BIO <-> spans, repair, span-level F1 (the contest metric)
  itn/cli.py            train | predict | run | evaluate | validate | selfcheck
  itn/transformer.py    optional XLM-R tagger (untested — see Status)
  tools/make_synthetic.py  data generator in the contest schema (for tests, not training)
  tests/test_pipeline.py
```
