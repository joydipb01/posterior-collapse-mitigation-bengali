"""Linear probes on the latent code.

This is the diagnostic that distinguishes the project from a plain benchmark:
instead of only reporting aggregate KL, we ask *which linguistic feature* the
latent still encodes after collapse.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import FEATURES, labels_for


@torch.no_grad()
def collect_latents(model, loader, device, max_batches: int | None = None):
    """Return posterior means and the matching source texts for a split."""
    model.eval()
    mus: list[torch.Tensor] = []
    texts: list[str] = []
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        mu, _ = model.encode(input_ids, lengths)
        mus.append(mu.cpu())
        texts.extend(batch["texts"])
    if not mus:
        return torch.empty(0), []
    return torch.cat(mus, dim=0), texts


def _standardise(train: torch.Tensor, *others: torch.Tensor) -> list[torch.Tensor]:
    mean = train.mean(0, keepdim=True)
    std = train.std(0, keepdim=True).clamp_min(1e-6)
    return [(x - mean) / std for x in (train, *others)]


def fit_linear_probe(
    z: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    *,
    epochs: int = 400,
    lr: float = 0.1,
    weight_decay: float = 1e-4,
    val_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, float]:
    """Fit logistic regression; return accuracy, balanced accuracy, macro-F1.

    Bengali probe labels are skewed (most sentences are third person, formal,
    present), so raw accuracy mostly measures the majority class.  Balanced
    accuracy and macro-F1 average over the classes that actually occur in the
    held-out split.
    """
    n = z.size(0)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    z, y = z[perm], y[perm]

    n_val = max(1, int(n * val_frac)) if n >= 10 else 0
    z_train, y_train = z[n_val:], y[n_val:]
    z_val, y_val = (z[:n_val], y[:n_val]) if n_val else (z, y)
    z_train, z_val = _standardise(z_train, z_val)

    model = nn.Linear(z.size(1), num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(z_train), y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = model(z_val).argmax(1)
        acc = (pred == y_val).float().mean().item()
        counts = torch.bincount(y_val, minlength=num_classes)
        baseline = (counts.max() / counts.sum()).item()

        recalls: list[float] = []
        f1s: list[float] = []
        for cls in torch.unique(y_val).tolist():
            tp = ((pred == cls) & (y_val == cls)).sum().item()
            fp = ((pred == cls) & (y_val != cls)).sum().item()
            fn = ((pred != cls) & (y_val == cls)).sum().item()
            recall = tp / (tp + fn) if tp + fn else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            recalls.append(recall)
            f1s.append(f1)

    return {
        "accuracy": acc,
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "majority": baseline,
        "n": int(n),
        "n_val": int(n_val),
    }


def probe_report(
    mu: torch.Tensor,
    texts: Sequence[str],
    gold: dict[str, dict[str, str]] | None = None,
    *,
    seed: int = 0,
) -> dict[str, dict]:
    """Probe each linguistic feature; skip features with <2 classes."""
    labels = labels_for(texts, gold)
    mu = mu.detach().cpu()
    report: dict[str, dict] = {}
    for feature in FEATURES:
        values = [row[feature] for row in labels]
        keep = [i for i, value in enumerate(values) if value is not None]
        classes = sorted({values[i] for i in keep})
        if len(keep) < 20 or len(classes) < 2:
            report[feature] = {
                "skipped": True,
                "n": len(keep),
                "classes": classes,
            }
            continue
        mapping = {name: idx for idx, name in enumerate(classes)}
        y = torch.tensor([mapping[values[i]] for i in keep], dtype=torch.long)
        result = fit_linear_probe(mu[keep], y, len(classes), seed=seed)
        result.update({"skipped": False, "classes": classes})
        report[feature] = result
    return report
