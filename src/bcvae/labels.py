"""Linguistic labels for the latent probes.

Two sources are supported:

* **gold** -- a TSV with ``text<TAB>tense<TAB>person<TAB>honorific`` (see
  ``data/README.md``);
* **silver** -- deterministic regex heuristics over Bengali pronouns and verb
  endings.  These are intentionally noisy and must be reported as silver
  labels in any write-up.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

FEATURES = ("tense", "person", "honorific")

_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "tense": [
        ("future", r"(বেন|বে|বো|বি|বা)"),
        ("past", r"(েছিলেন|েছিলাম|েছিল|ছিলেন|ছিলে|ছিল|লাম|লেন|লে|লো|লা)"),
        ("present", r"(চ্ছিলেন|চ্ছেন|চ্ছে|চ্ছি|েছেন|েছে|েছি|ছে|ছেন)"),
    ],
    "person": [
        ("1", r"(আমরা|আমাদেরকে|আমাদের|আমাকে|আমার|আমি)"),
        (
            "2",
            r"(আপনারা|আপনাদের|আপনাকে|আপনার|আপনি|তোমরা|তোমাদের|তোমাকে|তোমার|তুমি|তোরা|তোদের|তোর|তুই)",
        ),
        ("3", r"(তাঁরা|তাদেরকে|তাদের|তাঁকে|তাঁর|তিনি|তারা|তাকে|তার|সে|এরা|ওরা|উনি|ওঁরা|এটি|সেটি)"),
    ],
    "honorific": [
        ("formal", r"(আপনারা|আপনাদের|আপনাকে|আপনার|আপনি|তাঁরা|তাঁকে|তাঁর|তিনি|উনি|ওঁরা)"),
        ("informal", r"(তোমরা|তোমাদের|তোমাকে|তোমার|তুমি|তোরা|তোদের|তোর|তুই)"),
    ],
}

_COMPILED = {
    feature: [(label, re.compile(pattern)) for label, pattern in patterns]
    for feature, patterns in _PATTERNS.items()
}


def silver_labels(text: str) -> dict[str, str | None]:
    """Return the first heuristic match per feature (or ``None``)."""
    out: dict[str, str | None] = {}
    for feature, patterns in _COMPILED.items():
        out[feature] = next((label for label, rx in patterns if rx.search(text)), None)
    return out


def load_gold(path: str | Path) -> dict[str, dict[str, str]]:
    """Load ``text<TAB>tense<TAB>person<TAB>honorific``; header optional."""
    table: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"{path}:{lineno}: expected 4 tab-separated columns")
            if parts[0].lower() == "text" and parts[1].lower() == "tense":
                continue
            table[parts[0]] = dict(zip(FEATURES, (p.strip() or None for p in parts[1:]), strict=False))
    return table


def labels_for(
    texts: Iterable[str], gold: dict[str, dict[str, str]] | None = None
) -> list[dict[str, str | None]]:
    out = []
    for text in texts:
        row = silver_labels(text)
        if gold and text in gold:
            row = {**row, **gold[text]}
        out.append(row)
    return out
