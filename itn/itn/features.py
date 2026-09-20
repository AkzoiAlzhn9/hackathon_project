# -*- coding: utf-8 -*-
"""Feature extraction for the CRF tagger.

One feature list per token. The window is +-2 tokens plus adjacent bigrams,
which is what the class distinctions in this task actually hinge on: the same
word is CARDINAL, TIME, MEASURE or WHITELIST depending only on its neighbours
("говорил три раза" vs "в три тридцать" vs "через три минут").
"""

from functools import lru_cache
from typing import List, Sequence, Tuple

from . import lexicons

_PAD = "<pad>"
_OFFSETS = (-2, -1, 0, 1, 2)


@lru_cache(maxsize=1 << 18)
def _token_tag_string(token: str) -> str:
    tags = lexicons.tags(token)
    return "+".join(sorted(tags)) if tags else "-"


@lru_cache(maxsize=1 << 18)
def _token_tag_features(token: str) -> Tuple[str, ...]:
    return tuple(sorted(lexicons.tags(token)))


@lru_cache(maxsize=1 << 18)
def _affixes(word: str) -> Tuple[str, ...]:
    out = []
    for k in (2, 3, 4):
        if len(word) > k:
            out.append(f"suf{k}={word[-k:]}")
            out.append(f"pre{k}={word[:k]}")
    out.append(f"shape={_shape(word)}")
    return tuple(out)


def _shape(token: str) -> str:
    n = len(token)
    if n <= 2:
        return "s"
    if n <= 4:
        return "m"
    if n <= 7:
        return "l"
    return "xl"


def token_features(tokens: Sequence[str], i: int) -> List[str]:
    n = len(tokens)
    word = tokens[i]
    feats: List[str] = ["bias"]

    for offset in _OFFSETS:
        j = i + offset
        neighbour = tokens[j] if 0 <= j < n else _PAD
        feats.append(f"w[{offset}]={neighbour}")
        for tag in _token_tag_features(neighbour):
            feats.append(f"t[{offset}]={tag}")

    # Bigrams and a skip-gram: the deciding context is almost always adjacent.
    def w(offset: int) -> str:
        j = i + offset
        return tokens[j] if 0 <= j < n else _PAD

    feats.append(f"w[-1]|w[0]={w(-1)}|{w(0)}")
    feats.append(f"w[0]|w[1]={w(0)}|{w(1)}")
    feats.append(f"w[-2]|w[-1]={w(-2)}|{w(-1)}")
    feats.append(f"w[1]|w[2]={w(1)}|{w(2)}")
    feats.append(f"w[-1]|w[1]={w(-1)}|{w(1)}")

    # Tag bigrams generalise the above to unseen words.
    def tagstr(offset: int) -> str:
        return _token_tag_string(w(offset))

    feats.append(f"tt[-1]|tt[0]={tagstr(-1)}|{tagstr(0)}")
    feats.append(f"tt[0]|tt[1]={tagstr(0)}|{tagstr(1)}")
    feats.append(f"tt[-1]|tt[0]|tt[1]={tagstr(-1)}|{tagstr(0)}|{tagstr(1)}")

    # Sub-word shape: both languages inflect heavily, suffixes carry the case.
    feats.extend(_affixes(word))

    # Position.
    if i == 0:
        feats.append("BOS")
    if i == n - 1:
        feats.append("EOS")
    feats.append(f"pos={min(i, 5)}")
    feats.append(f"rpos={min(n - 1 - i, 5)}")
    feats.append(f"slen={min(n, 12)}")

    # Sentence-level cues: an email address is dictated with markers that may sit
    # several tokens away, and they are the only reliable EMAIL signal.
    if any(lexicons.is_email_context(t) for t in tokens):
        feats.append("sent_has_mail")
        left = any(lexicons.is_email_context(t) for t in tokens[:i])
        right = any(lexicons.is_email_context(t) for t in tokens[i + 1 :])
        feats.append(f"mail_side={int(left)}{int(right)}")
    if any(lexicons.is_number(t) for t in tokens):
        feats.append("sent_has_num")

    # Run length of consecutive numerals around the token: spans of numerals are
    # what the numeric classes are made of.
    if lexicons.is_number(word):
        left = 0
        j = i - 1
        while j >= 0 and lexicons.is_number(tokens[j]):
            left += 1
            j -= 1
        right = 0
        j = i + 1
        while j < n and lexicons.is_number(tokens[j]):
            right += 1
            j += 1
        feats.append(f"numrun={min(left, 4)}|{min(right, 4)}")

    return feats


def sentence_features(tokens: Sequence[str]) -> List[List[str]]:
    return [token_features(tokens, i) for i in range(len(tokens))]
