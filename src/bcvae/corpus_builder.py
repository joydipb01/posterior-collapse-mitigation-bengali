"""Streaming corpus builder for the Bengali VAE study.

The full FineWeb-2 Bengali split is ~87 GB, so this module streams documents
from the Hugging Face Hub rather than downloading them, splits each document
into sentences, filters noisy lines, de-duplicates, and writes a
**document-level** train/val/test split.

The command is intentionally conservative: it never mixes sentences from the
same document across splits, because sentence-level random splits leak context
between train and test and inflate probe accuracy.

Only ``datasets`` is optional.  Import it lazily so the rest of the package
works without the ``corpus`` extra.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .corpus import normalize_text, write_lines

BENGALI_START = 0x0980
BENGALI_END = 0x09FF

SCRIPT_RANGES: dict[str, tuple[int, int] | None] = {
    "bengali": (BENGALI_START, BENGALI_END),
    "latin": (0x0041, 0x024F),
    "any": None,
}

# A danda is an unambiguous sentence boundary, so it splits with or without a
# following space.  A full stop only splits when followed by whitespace and a
# letter, which keeps decimals such as 3.5 intact.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[।?!])\s*|(?<=\.)\s+(?=[A-Z\u0980-\u09FF])")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_HTML = re.compile(r"<[^>]{1,200}>")
_DEDUPE_STRIP = re.compile(r"[\W_]+", re.UNICODE)

SPLITS = ("train", "val", "test")


def split_sentences(text: str) -> list[str]:
    """Split a document into normalised Bengali sentences.

    This is a deterministic baseline, not a linguistic sentence splitter.  It
    handles danda, question marks, exclamation marks, newlines, and full stops
    followed by a letter.  Abbreviations are the main known failure mode.
    """
    out: list[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = normalize_text(paragraph)
        if not paragraph:
            continue
        for part in _SENTENCE_BOUNDARY.split(paragraph):
            part = normalize_text(part)
            if part:
                out.append(part)
    return out


def _script_characters(text: str) -> list[str]:
    """Letters plus combining marks (Bengali vowel signs are ``Mn``)."""
    return [
        ch
        for ch in text
        if ch.isalpha() or unicodedata.category(ch).startswith("M")
    ]


def script_ratio(text: str, script: str = "bengali") -> float:
    """Fraction of script characters that fall in the requested Unicode block."""
    chars = _script_characters(text)
    if not chars:
        return 0.0
    if script == "any":
        return 1.0
    try:
        start, end = SCRIPT_RANGES[script]
    except KeyError as exc:
        raise ValueError(
            f"unknown script {script!r}; choose one of {sorted(SCRIPT_RANGES)}"
        ) from exc
    return sum(start <= ord(ch) <= end for ch in chars) / len(chars)


def bengali_ratio(text: str) -> float:
    """Backwards-compatible wrapper for the Bengali script ratio."""
    return script_ratio(text, "bengali")


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch.isdigit() for ch in text) / len(text)


def passes_filters(text: str, cfg: Config) -> tuple[bool, str]:
    """Apply cheap quality filters; return ``(keep, rejection_reason)``."""
    min_chars = int(cfg.get_path("min_chars", 20))
    max_chars = int(cfg.get_path("max_chars", 200))
    min_words = int(cfg.get_path("min_words", 4))
    max_words = int(cfg.get_path("max_words", 40))
    script = str(cfg.get_path("script", "bengali")).lower()
    min_script_ratio = float(
        cfg.get_path("min_script_ratio", cfg.get_path("min_bengali_ratio", 0.8))
    )
    max_digit_ratio = float(cfg.get_path("max_digit_ratio", 0.2))
    require_terminal = bool(cfg.get_path("require_terminal_punctuation", False))

    if not (min_chars <= len(text) <= max_chars):
        return False, "length"

    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return False, "words"

    if script_ratio(text, script) < min_script_ratio:
        return False, "script"

    if digit_ratio(text) > max_digit_ratio:
        return False, "digits"

    if _URL.search(text) or _EMAIL.search(text) or _HTML.search(text):
        return False, "noise"

    if require_terminal and not text.endswith(("।", "?", "!", ".")):
        return False, "punctuation"

    return True, ""


def assign_split(doc_id: str, val_frac: float, test_frac: float, *, seed: int = 13) -> str:
    """Assign a whole document to a split with a stable hash."""
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac/test_frac must be in [0, 1) and sum to < 1")
    payload = f"{seed}:{doc_id}".encode()
    bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / 2**64
    if bucket < val_frac:
        return "val"
    if bucket < val_frac + test_frac:
        return "test"
    return "train"


def _document_id(doc: dict[str, Any], id_field: str | None, text_field: str) -> str:
    raw = doc.get(id_field) if id_field else None
    if raw is None or raw == "":
        raw = doc.get(text_field, "")
    return hashlib.blake2b(str(raw).encode("utf-8"), digest_size=16).hexdigest()


def _dedupe_key(text: str, mode: str) -> str | None:
    if mode == "none":
        return None
    if mode == "exact":
        return text
    return _DEDUPE_STRIP.sub("", text).lower()


@dataclass(frozen=True)
class SourceSpec:
    """Resolved description of where documents come from."""

    kind: str
    dataset: str | None
    config: str | None
    split: str
    path: str | None
    language: str | None
    language_field: str
    text_field: str
    id_field: str | None
    revision: str | None


def resolve_source(cfg: Config) -> SourceSpec:
    """Resolve aliases and defaults into a concrete source description."""
    kind = str(cfg.get_path("source", "fineweb2")).lower()
    dataset = cfg.get_path("dataset")
    config = cfg.get_path("config")
    split = str(cfg.get_path("split", "train"))
    path = cfg.get_path("path")
    language = cfg.get_path("language")
    language_field = str(cfg.get_path("language_field", "language"))
    text_field = str(cfg.get_path("text_field", "text"))
    id_field = cfg.get_path("id_field") or None
    revision = cfg.get_path("revision") or None

    if kind == "fineweb2":
        if not dataset or dataset == "ai4bharat/sangraha":
            dataset = "HuggingFaceFW/fineweb-2"
        if not config or config == "verified":
            config = "ben_Beng"
    elif kind == "sangraha":
        if not dataset or dataset == "HuggingFaceFW/fineweb-2":
            dataset = "ai4bharat/sangraha"
        if not config or config == "ben_Beng":
            config = "verified"
        # Sangraha exposes one split per language (e.g. split="ben"), not a
        # single train split with a `language` column.
        if split == "train":
            split = "ben"
        language = None
    elif kind == "hf":
        if not dataset:
            raise ValueError("source: hf needs a 'dataset'")
    elif kind == "jsonl":
        if not path:
            raise ValueError("source: jsonl needs a 'path'")
    else:
        raise ValueError(f"unknown corpus source: {kind!r}")

    return SourceSpec(
        kind=kind,
        dataset=dataset,
        config=config,
        split=split,
        path=path,
        language=language,
        language_field=language_field,
        text_field=text_field,
        id_field=id_field,
        revision=revision,
    )


def _iter_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            yield row


def _load_hf(spec: SourceSpec):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised by users, not tests
        raise ImportError(
            "The corpus builder needs the optional 'datasets' dependency.\n"
            "Install it with:  uv pip install -e \".[corpus]\""
        ) from exc

    kwargs: dict[str, Any] = {"split": spec.split, "streaming": True}
    if spec.revision:
        kwargs["revision"] = spec.revision
    if spec.config:
        dataset = load_dataset(spec.dataset, spec.config, **kwargs)
    else:
        dataset = load_dataset(spec.dataset, **kwargs)

    if spec.language:
        wanted = spec.language.lower()
        dataset = dataset.filter(
            lambda row: str(row.get(spec.language_field, "")).lower() == wanted
        )
    return dataset


def iter_documents(spec: SourceSpec) -> Iterator[dict[str, Any]]:
    if spec.kind == "jsonl":
        yield from _iter_jsonl(spec.path or "")
    else:
        yield from _load_hf(spec)


def _filter_settings(cfg: Config) -> dict[str, Any]:
    return {
        "min_chars": int(cfg.get_path("min_chars", 20)),
        "max_chars": int(cfg.get_path("max_chars", 200)),
        "min_words": int(cfg.get_path("min_words", 4)),
        "max_words": int(cfg.get_path("max_words", 40)),
        "script": str(cfg.get_path("script", "bengali")).lower(),
        "min_script_ratio": float(
            cfg.get_path("min_script_ratio", cfg.get_path("min_bengali_ratio", 0.8))
        ),
        "max_digit_ratio": float(cfg.get_path("max_digit_ratio", 0.2)),
        "require_terminal_punctuation": bool(
            cfg.get_path("require_terminal_punctuation", False)
        ),
    }


def _print(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, flush=True)


def build_corpus(
    cfg: Config,
    *,
    dry_run: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Stream, filter, and write ``train/val/test.txt`` plus ``manifest.json``."""
    spec = resolve_source(cfg)
    out = Path(cfg.get_path("out", "data/processed"))

    targets = {
        "train": int(cfg.get_path("max_sentences", 2_000_000)),
        "val": int(cfg.get_path("val_sentences", 20_000)),
        "test": int(cfg.get_path("test_sentences", 20_000)),
    }
    if any(value < 0 for value in targets.values()):
        raise ValueError("sentence targets must be non-negative")
    if targets["train"] == 0:
        raise ValueError("max_sentences must be > 0")

    val_frac = float(cfg.get_path("val_frac", 0.01))
    test_frac = float(cfg.get_path("test_frac", 0.01))
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac/test_frac must be in [0, 1) and sum to < 1")
    seed = int(cfg.get_path("seed", 13))
    dedupe = str(cfg.get_path("dedupe", "normalized")).lower()
    if dedupe not in {"exact", "normalized", "none"}:
        raise ValueError("dedupe must be one of: exact, normalized, none")

    max_documents = cfg.get_path("max_documents")
    max_documents = int(max_documents) if max_documents is not None else None
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be > 0")
    show_progress = bool(cfg.get_path("show_progress", True)) and not quiet
    filters = _filter_settings(cfg)

    plan = {
        "source": asdict(spec),
        "out": str(out),
        "targets": targets,
        "split": {
            "val_frac": val_frac,
            "test_frac": test_frac,
            "seed": seed,
            "method": "blake2b(document_id)",
        },
        "filters": filters,
        "dedupe": dedupe,
        "max_documents": max_documents,
    }
    if dry_run:
        _print(json.dumps(plan, ensure_ascii=False, indent=2), quiet)
        return plan

    texts: dict[str, list[str]] = {split: [] for split in SPLITS}
    seen: set[str] = set()
    counts: dict[str, Any] = {
        "documents_scanned": 0,
        "documents_used": 0,
        "sentences_seen": 0,
        "sentences_kept": 0,
        "duplicates": 0,
        "filtered": 0,
    }
    for reason in ("length", "words", "script", "digits", "noise", "punctuation"):
        counts[f"filtered_{reason}"] = 0

    documents = iter_documents(spec)
    if show_progress:
        from tqdm import tqdm

        documents = tqdm(documents, desc="documents", unit="doc")

    for doc in documents:
        if max_documents is not None and counts["documents_scanned"] >= max_documents:
            break
        counts["documents_scanned"] += 1
        if all(len(texts[split]) >= targets[split] for split in SPLITS):
            break

        doc_id = _document_id(doc, spec.id_field, spec.text_field)
        split = assign_split(doc_id, val_frac, test_frac, seed=seed)
        if len(texts[split]) >= targets[split]:
            continue

        raw = doc.get(spec.text_field)
        if not isinstance(raw, str):
            counts["filtered"] += 1
            counts["filtered_noise"] += 1
            continue

        kept_this_document = 0
        for sentence in split_sentences(raw):
            counts["sentences_seen"] += 1
            if len(texts[split]) >= targets[split]:
                break

            keep, reason = passes_filters(sentence, cfg)
            if not keep:
                counts["filtered"] += 1
                counts[f"filtered_{reason}"] += 1
                continue

            key = _dedupe_key(sentence, dedupe)
            if key is not None:
                if key in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(key)

            texts[split].append(sentence)
            kept_this_document += 1

        if kept_this_document:
            counts["documents_used"] += 1

    counts["sentences_kept"] = sum(len(rows) for rows in texts.values())
    counts["per_split"] = {split: len(texts[split]) for split in SPLITS}

    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = out / f"{split}.txt"
        write_lines(path, texts[split])
        files[split] = {
            "path": str(path),
            "sentences": len(texts[split]),
            "target": targets[split],
        }

    warnings = [
        f"{split}: kept {len(texts[split])} of {targets[split]} target sentences"
        for split in SPLITS
        if len(texts[split]) < targets[split]
    ]

    manifest: dict[str, Any] = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": asdict(spec),
        "license": cfg.get_path("license"),
        "attribution": cfg.get_path("attribution"),
        "targets": targets,
        "filters": filters,
        "dedupe": dedupe,
        "split": plan["split"],
        "max_documents": max_documents,
        "counts": counts,
        "files": files,
        "warnings": warnings,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "out": str(out),
        "manifest": str(manifest_path),
        "files": files,
        "documents_scanned": counts["documents_scanned"],
        "documents_used": counts["documents_used"],
        "sentences_kept": counts["sentences_kept"],
        "duplicates": counts["duplicates"],
        "filtered": counts["filtered"],
        "warnings": warnings,
    }
    _print(json.dumps(summary, ensure_ascii=False, indent=2), quiet)
    return manifest
