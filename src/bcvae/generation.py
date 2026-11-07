"""Latent -> text helpers shared by evaluation and interpolation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .models import SentenceVAE
from .tokenization import Tokenizer


@torch.no_grad()
def encode_text(
    model: SentenceVAE,
    tokenizer: Tokenizer,
    text: str,
    *,
    max_tokens: int = 40,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    ids = tokenizer.encode(text)[: max(1, max_tokens - 2)]
    input_ids = torch.tensor([[tokenizer.bos_id, *ids, tokenizer.eos_id]], device=device)
    lengths = torch.tensor([input_ids.size(1)], device=device)
    mu, _ = model.encode(input_ids, lengths)
    return mu.squeeze(0)


@torch.no_grad()
def decode_from_z(
    model: SentenceVAE,
    z: torch.Tensor,
    *,
    bos_id: int,
    eos_id: int,
    max_len: int = 40,
    greedy: bool = True,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Decode a batch of latents; returns token ids including BOS."""
    was_training = model.training
    model.eval()
    batch = z.size(0)
    ids = torch.full((batch, 1), bos_id, dtype=torch.long, device=z.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=z.device)

    for _ in range(max_len):
        logits = model.decoder(z, ids, None)[:, -1]
        if greedy:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(1)
        nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        finished |= nxt.eq(eos_id)
        if bool(finished.all()):
            break

    model.train(was_training)
    return ids


def ids_to_text(tokenizer: Tokenizer, ids: Sequence[int] | torch.Tensor) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return tokenizer.decode(list(ids))
