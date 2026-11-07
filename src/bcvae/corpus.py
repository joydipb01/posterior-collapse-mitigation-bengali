"""Corpus reading, Bengali-aware normalisation, and splitting."""

from __future__ import annotations

import random
import re
import unicodedata
from pathlib import Path

_WHITESPACE = re.compile(r"\s+")

# ZWNJ / ZWJ are meaningful in Bengali conjuncts; everything else in Unicode
# category C (control, format, surrogate, private-use) is dropped.
_KEEP_FORMAT = {"\u200c", "\u200d"}


def normalize_text(text: str) -> str:
    """NFC-normalise, drop control characters, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = "".join(
        ch
        for ch in text
        if ch in _KEEP_FORMAT or not unicodedata.category(ch).startswith("C")
    )
    return _WHITESPACE.sub(" ", text).strip()


def read_sentences(
    path: str | Path,
    *,
    max_sentences: int | None = None,
    min_chars: int = 2,
) -> list[str]:
    """Read one sentence per line, normalising and de-duplicating."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"corpus not found: {path}\n"
            "See data/README.md for the expected layout, or run with "
            "configs/smoke.yaml to use the bundled sample."
        )
    seen: set[str] = set()
    out: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = normalize_text(line)
            if len(text) < min_chars or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if max_sentences and len(out) >= max_sentences:
                break
    return out


def split_corpus(
    sentences: list[str],
    *,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 13,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministic train/val/test split."""
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac/test_frac must be in [0, 1) and sum to < 1")
    data = list(sentences)
    random.Random(seed).shuffle(data)
    n = len(data)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = data[:n_val]
    test = data[n_val : n_val + n_test]
    train = data[n_val + n_test :]
    return train, val, test


def write_lines(path: str | Path, lines: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
