"""Collapse diagnostics computed from the posterior."""

from __future__ import annotations

import numpy as np


def per_dim_kl(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    """KL(q(z|x) || N(0, I)) averaged over the batch, per dimension."""
    return (-0.5 * (1 + logvar - mu**2 - np.exp(logvar))).mean(axis=0)


def active_units(kl_per_dim: np.ndarray, threshold: float = 1e-2) -> int:
    """Number of latent dimensions carrying more than ``threshold`` nats."""
    return int((kl_per_dim > threshold).sum())


def effective_dim(kl_per_dim: np.ndarray) -> float:
    """Participation ratio of the per-dimension KL; 1 = fully collapsed."""
    kl = np.clip(kl_per_dim, 0, None)
    denom = float((kl**2).sum())
    if denom <= 0:
        return 0.0
    return float(kl.sum() ** 2 / denom)


def summarise(mu: np.ndarray, logvar: np.ndarray) -> dict[str, float]:
    kl = per_dim_kl(mu, logvar)
    return {
        "kl": float(kl.sum()),
        "active_units": active_units(kl),
        "effective_dim": effective_dim(kl),
    }
