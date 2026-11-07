"""Tokenizers: character, word, BPE, and a heuristic morpheme segmenter.

Every tokenizer exposes the same small interface and is serialised to a
single ``tokenizer.json`` so a run directory is self-contained.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class Tokenizer(ABC):
    type_name: str = "base"
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str: ...

    def save(self, path: str | Path) -> None:
        save_tokenizer(self, path)


class _VocabTokenizer(Tokenizer):
    """Shared logic for the tokenizers with an explicit ``itos`` list."""

    type_name = "vocab"

    def __init__(self, itos: list[str], meta: dict | None = None) -> None:
        self.itos = itos
        self.stoi = {token: idx for idx, token in enumerate(itos)}
        self.meta = meta or {}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def decode(self, ids: Sequence[int]) -> str:
        tokens = [
            self.itos[i]
            for i in ids
            if 0 <= i < len(self.itos) and i not in (self.pad_id, self.bos_id, self.eos_id)
        ]
        return self._detokenize(tokens)

    def _detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class CharTokenizer(_VocabTokenizer):
    type_name = "char"

    @classmethod
    def train(
        cls, sentences: Iterable[str], vocab_size: int | None = None, min_frequency: int = 1
    ) -> CharTokenizer:
        counts = Counter(ch for text in sentences for ch in text)
        chars = [ch for ch, n in counts.most_common() if n >= min_frequency]
        if vocab_size:
            chars = chars[: max(1, vocab_size - len(SPECIAL_TOKENS))]
        return cls(SPECIAL_TOKENS + chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def _detokenize(self, tokens: list[str]) -> str:
        # Spaces are part of the character vocabulary, so no separator is added.
        return "".join(tokens)


class WordTokenizer(_VocabTokenizer):
    type_name = "word"

    @classmethod
    def train(
        cls, sentences: Iterable[str], vocab_size: int = 8000, min_frequency: int = 1
    ) -> WordTokenizer:
        counts = Counter(word for text in sentences for word in text.split())
        words = [w for w, n in counts.most_common() if n >= min_frequency]
        words = words[: max(1, vocab_size - len(SPECIAL_TOKENS))]
        return cls(SPECIAL_TOKENS + words)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(word, self.unk_id) for word in text.split()]


# Longest-match first.  This is a baseline, not a linguistic analyser:
# replace it with BanglaMorph / Morfessor output for the morphology study.
DEFAULT_SUFFIXES = [
    "গুলোকে", "গুলোর", "গুলিতে", "গুলির", "গুলো", "গুলি",
    "দেরকে", "দের", "খানা", "খানি", "টার", "টির", "টাকে", "টিতে",
    "ছিলাম", "ছিলেন", "ছিলে", "ছিল", "চ্ছিলেন", "চ্ছিলাম", "চ্ছিল",
    "চ্ছেন", "চ্ছে", "চ্ছি", "েছেন", "েছে", "েছি", "েছিলেন", "েছিল",
    "বেন", "বে", "বো", "বি", "বা", "লাম", "লেন", "লে", "লো", "লা",
    "ের", "এর", "তে", "কে", "রা", "টা", "টি", "টা", "র", "ে", "ো", "ি", "া",
]


class MorphAnalyzer:
    """Greedy suffix stripper (at most two suffixes, stem >= ``min_stem``)."""

    def __init__(self, suffixes: Sequence[str] | None = None, min_stem: int = 2) -> None:
        self.suffixes = sorted(set(suffixes or DEFAULT_SUFFIXES), key=len, reverse=True)
        self.min_stem = min_stem

    def segment(self, word: str) -> list[str]:
        pieces: list[str] = []
        rest = word
        for _ in range(2):
            if len(rest) < self.min_stem + 1:
                break
            for suffix in self.suffixes:
                if rest.endswith(suffix) and len(rest) - len(suffix) >= self.min_stem:
                    pieces.insert(0, suffix)
                    rest = rest[: -len(suffix)]
                    break
            else:
                break
        pieces.insert(0, rest)
        return pieces


class MorphTokenizer(_VocabTokenizer):
    type_name = "morph"

    def __init__(
        self, itos: list[str], meta: dict | None = None, suffixes: Sequence[str] | None = None
    ) -> None:
        super().__init__(itos, meta)
        self.analyzer = MorphAnalyzer(suffixes)

    @classmethod
    def train(
        cls,
        sentences: Iterable[str],
        vocab_size: int = 8000,
        min_frequency: int = 1,
        suffixes: Sequence[str] | None = None,
    ) -> MorphTokenizer:
        analyzer = MorphAnalyzer(suffixes)
        counts: Counter[str] = Counter()
        for text in sentences:
            for word in text.split():
                for i, piece in enumerate(analyzer.segment(word)):
                    counts[("▁" if i == 0 else "##") + piece] += 1
        pieces = [p for p, n in counts.most_common() if n >= min_frequency]
        pieces = pieces[: max(1, vocab_size - len(SPECIAL_TOKENS))]
        return cls(SPECIAL_TOKENS + pieces, {"suffixes": analyzer.suffixes}, suffixes)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            for i, piece in enumerate(self.analyzer.segment(word)):
                token = ("▁" if i == 0 else "##") + piece
                ids.append(self.stoi.get(token, self.unk_id))
        return ids

    def _detokenize(self, tokens: list[str]) -> str:
        out: list[str] = []
        for token in tokens:
            if token.startswith("▁"):
                out.append(" " + token[1:])
            elif token.startswith("##"):
                out.append(token[2:])
            else:
                out.append(token)
        return "".join(out).strip()


class BPETokenizer(Tokenizer):
    type_name = "bpe"

    def __init__(self, hf_tokenizer, meta: dict | None = None) -> None:
        self.tokenizer = hf_tokenizer
        self.meta = meta or {}
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @classmethod
    def train(
        cls,
        sentences: Iterable[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> BPETokenizer:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers import decoders, models, pre_tokenizers, trainers

        tok = HFTokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(list(sentences), trainer=trainer)
        return cls(tok)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id}
        ids = [i for i in ids if i not in specials]
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def build_tokenizer(
    kind: str,
    sentences: Iterable[str],
    *,
    vocab_size: int = 8000,
    min_frequency: int = 2,
    morph_suffixes: Sequence[str] | None = None,
) -> Tokenizer:
    kind = kind.lower()
    if kind == "char":
        return CharTokenizer.train(sentences, vocab_size, min_frequency)
    if kind == "word":
        return WordTokenizer.train(sentences, vocab_size, min_frequency)
    if kind == "bpe":
        return BPETokenizer.train(sentences, vocab_size, min_frequency)
    if kind == "morph":
        return MorphTokenizer.train(sentences, vocab_size, min_frequency, morph_suffixes)
    raise ValueError(f"unknown tokenizer type: {kind!r}")


def save_tokenizer(tokenizer: Tokenizer, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if isinstance(tokenizer, BPETokenizer):
        payload = {
            "type": "bpe",
            "tokenizer": json.loads(tokenizer.tokenizer.to_str()),
        }
    elif isinstance(tokenizer, _VocabTokenizer):
        payload = {
            "type": tokenizer.type_name,
            "itos": tokenizer.itos,
            "meta": tokenizer.meta,
        }
    else:  # pragma: no cover - defensive
        raise TypeError(f"cannot serialise {type(tokenizer)!r}")
    (path / "tokenizer.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def load_tokenizer(path: str | Path) -> Tokenizer:
    payload = json.loads((Path(path) / "tokenizer.json").read_text(encoding="utf-8"))
    kind = payload["type"]
    if kind == "bpe":
        from tokenizers import Tokenizer as HFTokenizer

        tok = HFTokenizer.from_str(json.dumps(payload["tokenizer"]))
        return BPETokenizer(tok)
    if kind == "char":
        return CharTokenizer(payload["itos"], payload.get("meta"))
    if kind == "word":
        return WordTokenizer(payload["itos"], payload.get("meta"))
    if kind == "morph":
        return MorphTokenizer(payload["itos"], payload.get("meta"), payload.get("meta", {}).get("suffixes"))
    raise ValueError(f"unknown serialised tokenizer type: {kind!r}")
