"""PyTorch dataset and collate function."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from .tokenization import Tokenizer


class SentenceDataset(Dataset):
    """Tokenises once up front; returns BOS + ids + EOS sequences."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: Tokenizer,
        max_tokens: int = 40,
        min_tokens: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.items: list[dict[str, Any]] = []
        for text in texts:
            ids = tokenizer.encode(text)[: max(1, max_tokens - 2)]
            if len(ids) < min_tokens:
                continue
            ids = [tokenizer.bos_id, *ids, tokenizer.eos_id]
            self.items.append({"input_ids": ids, "text": text})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def collate_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    lengths = torch.empty(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        ids = torch.tensor(item["input_ids"], dtype=torch.long)
        input_ids[i, : len(ids)] = ids
        lengths[i] = len(ids)
    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "texts": [item["text"] for item in batch],
    }
