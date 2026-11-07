"""Latent interpolation and chain-level metrics."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch

from ..generation import decode_from_z, encode_text, ids_to_text
from ..models import SentenceVAE
from ..tokenization import Tokenizer
from .text_metrics import char_ngram_similarity, distinct_n, spearman, token_f1


def _slerp(z0: torch.Tensor, z1: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    n0 = z0.norm().clamp_min(eps)
    n1 = z1.norm().clamp_min(eps)
    a, b = z0 / n0, z1 / n1
    dot = (a * b).sum().clamp(-1 + eps, 1 - eps)
    omega = torch.acos(dot)
    if float(omega) < 1e-3:
        return torch.stack([(1 - t) * z0 + t * z1 for t in ts])
    sin_omega = torch.sin(omega)
    w0 = torch.sin((1 - ts) * omega) / sin_omega
    w1 = torch.sin(ts * omega) / sin_omega
    return w0[:, None] * z0[None, :] + w1[:, None] * z1[None, :]


def interpolate_latents(
    mu_a: torch.Tensor, mu_b: torch.Tensor, steps: int = 7, mode: str = "linear"
) -> torch.Tensor:
    """Return ``steps + 2`` latents, endpoints included."""
    ts = torch.linspace(0, 1, steps + 2, device=mu_a.device)
    if mode == "linear":
        return torch.stack([(1 - t) * mu_a + t * mu_b for t in ts])
    if mode == "slerp":
        return _slerp(mu_a, mu_b, ts)
    raise ValueError(f"unknown interpolation mode: {mode!r}")


@torch.no_grad()
def interpolate_chain(
    model: SentenceVAE,
    tokenizer: Tokenizer,
    text_a: str,
    text_b: str,
    *,
    steps: int = 7,
    mode: str = "linear",
    device: torch.device | str = "cpu",
    max_tokens: int = 40,
    greedy: bool = True,
) -> list[str]:
    mu_a = encode_text(model, tokenizer, text_a, max_tokens=max_tokens, device=device)
    mu_b = encode_text(model, tokenizer, text_b, max_tokens=max_tokens, device=device)
    zs = interpolate_latents(mu_a, mu_b, steps, mode)
    ids = decode_from_z(
        model,
        zs,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        max_len=max_tokens,
        greedy=greedy,
    )
    return [ids_to_text(tokenizer, row) for row in ids]


def chain_metrics(texts: Sequence[str]) -> dict:
    """Endpoint overlap, step smoothness, monotonicity, and a jump statistic."""
    if len(texts) < 2:
        raise ValueError("a chain needs at least two sentences")
    to_start = [token_f1(t, texts[0]) for t in texts]
    to_end = [token_f1(t, texts[-1]) for t in texts]
    sims = [
        char_ngram_similarity(texts[i], texts[i + 1]) for i in range(len(texts) - 1)
    ]
    jumps = [1 - s for s in sims]
    return {
        "n_steps": len(texts),
        "f1_to_start": to_start,
        "f1_to_end": to_end,
        "step_similarity": sims,
        "smoothness": sum(sims) / len(sims),
        "max_jump": max(jumps),
        "monotonicity": spearman(list(range(len(to_end))), to_end),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
    }


def read_pairs(path: str | Path) -> list[tuple[str, str]]:
    pairs = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|||")]
            if len(parts) != 2 or not all(parts):
                raise ValueError(f"{path}:{lineno}: expected 'sentence A ||| sentence B'")
            pairs.append((parts[0], parts[1]))
    return pairs


@torch.no_grad()
def interpolate_pairs(
    model: SentenceVAE,
    tokenizer: Tokenizer,
    pairs: Sequence[tuple[str, str]],
    *,
    steps: int = 7,
    mode: str = "linear",
    device: torch.device | str = "cpu",
    max_tokens: int = 40,
) -> list[dict]:
    results = []
    for a, b in pairs:
        texts = interpolate_chain(
            model,
            tokenizer,
            a,
            b,
            steps=steps,
            mode=mode,
            device=device,
            max_tokens=max_tokens,
        )
        results.append(
            {
                "source": a,
                "target": b,
                "mode": mode,
                "chain": texts,
                "metrics": chain_metrics(texts),
            }
        )
    return results


def write_results(results: Sequence[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
