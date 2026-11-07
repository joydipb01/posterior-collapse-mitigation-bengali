"""KL annealing schedules (pure functions, unit-tested)."""

from __future__ import annotations


def beta_at(
    step: int,
    *,
    schedule: str = "linear",
    warmup_steps: int = 0,
    total_steps: int = 1,
    cycles: int = 4,
    ratio: float = 0.5,
) -> float:
    """Return the KL weight for a global training step.

    ``linear``   -- 0 -> 1 over ``warmup_steps`` (Bowman et al., 2016).
    ``cyclical`` -- sawtooth with ``cycles`` periods (Fu et al., 2019).
    ``constant`` -- always 1.
    """
    if schedule == "constant":
        return 1.0
    if schedule == "linear":
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, step / warmup_steps)
    if schedule == "cyclical":
        cycle = max(1, total_steps // max(1, cycles))
        rise = max(1, int(ratio * cycle))
        return min(1.0, (step % cycle) / rise)
    raise ValueError(f"unknown KL schedule: {schedule!r}")
