"""Learning-rate schedule factories."""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from flatbuild.config import SchedulerConfig


def build_scheduler(
    config: SchedulerConfig,
    optimizer: Optimizer,
    *,
    total_steps: int | None = None,
) -> LambdaLR:
    """Construct an LR scheduler.

    Args:
        config: Schedule configuration.
        optimizer: Optimizer whose parameter groups will be scheduled.
        total_steps: Number of training steps over which the schedule
            applies. When ``None``, the schedule is open-ended.

    Returns:
        A configured :class:`LambdaLR`.
    """
    name = (config.type or "cosine").lower()
    if name == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    if name == "cosine":
        warmup = max(0, int(config.warmup_steps))
        min_ratio = float(config.min_lr_ratio)
        total = max(1, int(total_steps or 0))
        return LambdaLR(
            optimizer,
            lr_lambda=lambda step: _cosine_lr(step, warmup, total, min_ratio),
        )
    raise ValueError(f"Unknown scheduler type: {config.type!r}")


def _cosine_lr(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Compute lr multiplier at ``step`` for cosine schedule + warmup."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
