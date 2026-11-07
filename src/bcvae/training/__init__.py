"""Training helpers.

Import ``train`` from ``bcvae.training.trainer`` directly to avoid a circular
import with the evaluation package (trainer -> eval.reconstruct -> metrics).
"""

from .schedules import beta_at

__all__ = ["beta_at"]
