"""Optimizer factories.

Currently only AdamW is supported. The factory exists so future
optimizers (Lion, Adafactor, …) plug in without touching the trainer.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from flatbuild.config import OptimizerConfig


def build_optimizer(
    config: OptimizerConfig,
    parameters: Iterable[nn.Parameter],
) -> torch.optim.Optimizer:
    """Construct an optimizer from a config.

    Args:
        config: Optimizer hyperparameters.
        parameters: Model parameters (typically
            ``model.parameters()``).

    Returns:
        A configured :class:`torch.optim.Optimizer`.

    Raises:
        ValueError: If the optimizer type is unknown.
    """
    params_list = [p for p in parameters if p.requires_grad]
    if not params_list:
        # Empty models still get a valid optimizer; trainer will fail fast
        # before reaching the first step.
        params_list = list(parameters)

    name = (config.type or "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(
            params_list,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            params_list,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(
            params_list,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer type: {config.type!r}")
