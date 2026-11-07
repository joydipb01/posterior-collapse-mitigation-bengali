"""Reconstruction and collapse evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from ..generation import decode_from_z, ids_to_text
from ..models import SentenceVAE
from ..tokenization import Tokenizer
from ..training.metrics import active_units, effective_dim
from .text_metrics import token_f1


@torch.no_grad()
def evaluate(
    model: SentenceVAE,
    loader: Iterable[dict],
    tokenizer: Tokenizer,
    device: torch.device,
    *,
    beta: float = 1.0,
    max_batches: int | None = None,
    sample_n: int = 4,
    max_len: int = 40,
) -> dict:
    model.eval()
    nll_sum = 0.0
    n_tokens = 0.0
    kl_sum: np.ndarray | None = None
    n_sentences = 0
    samples: list[dict[str, str]] = []

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        out = model(input_ids, lengths, beta=beta)

        nll_sum += float(out["nll_per_token"]) * float(out["n_tokens"])
        n_tokens += float(out["n_tokens"])
        kl = out["kl_per_dim"].cpu().numpy()
        kl_sum = kl if kl_sum is None else kl_sum + kl * input_ids.size(0)
        n_sentences += input_ids.size(0)

        if len(samples) < sample_n:
            ids = decode_from_z(
                model,
                out["mu"],
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_len=max_len,
            )
            for ref, hyp in zip(batch["texts"], ids, strict=True):
                samples.append(
                    {
                        "reference": ref,
                        "reconstruction": ids_to_text(tokenizer, hyp),
                        "f1": round(token_f1(ids_to_text(tokenizer, hyp), ref), 4),
                    }
                )

    kl_per_dim = (kl_sum / max(1, n_sentences)) if kl_sum is not None else np.zeros(1)
    nll = nll_sum / max(1.0, n_tokens)
    return {
        "nll_per_token": nll,
        "perplexity": float(np.exp(nll)),
        "kl": float(kl_per_dim.sum()),
        "active_units": active_units(kl_per_dim),
        "effective_dim": effective_dim(kl_per_dim),
        "n_sentences": n_sentences,
        "samples": samples,
    }
