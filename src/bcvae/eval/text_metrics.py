"""Dependency-free text metrics.

Char n-gram cosine is the default semantic-ish proxy because Bengali has free
word order and BLEU-style overlap punishes valid reorderings.  ``sacrebleu``
and ``bert-score`` are optional extras.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np


def token_f1(prediction: str, reference: str) -> float:
    pred = Counter(prediction.split())
    ref = Counter(reference.split())
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(pred.values()))
    recall = overlap / max(1, sum(ref.values()))
    return 2 * precision * recall / (precision + recall)


def _char_ngrams(text: str, n: int = 3) -> Counter:
    text = "".join(text.split())
    if len(text) < n:
        return Counter({text: 1})
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def char_ngram_similarity(a: str, b: str, n: int = 3) -> float:
    ca, cb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not ca or not cb:
        return 0.0
    dot = sum((ca & cb).values())
    norm = np.sqrt(sum(ca.values()) * sum(cb.values()))
    return float(dot / norm) if norm else 0.0


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    grams: list[tuple] = []
    for text in texts:
        tokens = text.split()
        grams.extend(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float | None:
    try:
        import sacrebleu
    except ImportError:
        return None
    result = sacrebleu.corpus_bleu(
        list(hypotheses), [list(references)], tokenize="char"
    )
    return float(result.score)
