"""Gradient clipping callback."""

from __future__ import annotations

import torch

from flatbuild.callbacks.base import Callback, CallbackContext


class GradientClipCallback(Callback):
    """Clip gradient norms to ``max_norm`` before each optimizer step.

    Args:
        max_norm: Max norm (default ``1.0``).
        norm_type: p for the norm (default ``2`` — L2).
    """

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0) -> None:
        """Initialize.

        Args:
            max_norm: Maximum allowed gradient norm.
            norm_type: Norm type passed to :func:`torch.nn.utils.clip_grad_norm_`.
        """
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_step_begin(self, ctx: CallbackContext) -> None:
        """Clip gradients for the trainer's parameters."""
        torch.nn.utils.clip_grad_norm_(
            ctx.trainer.model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type,
        )
