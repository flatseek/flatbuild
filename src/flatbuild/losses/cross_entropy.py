"""Cross-entropy losses for language modeling."""

from __future__ import annotations

import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """Shifted-token cross-entropy (next-token prediction).

    Wraps :func:`torch.nn.functional.cross_entropy` with the
    autoregressive shift applied to ``logits`` / ``labels``.
    """

    def __init__(self, ignore_index: int = -100) -> None:
        """Initialize the loss.

        Args:
            ignore_index: Label value to ignore (typically ``-100`` for
                masked positions, e.g. the prompt part of a conversation).
        """
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits: Tensor of shape ``(B, T, V)``.
            labels: Tensor of shape ``(B, T)``.

        Returns:
            Scalar loss.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
        )


class LabelSmoothingLoss(nn.Module):
    """Label-smoothed cross-entropy (Szegedy et al., 2016).

    Equivalent to standard CE plus a per-token ``eps * uniform``
    regularizer on the prediction distribution. Useful for very small
    datasets like the demo conversational model.
    """

    def __init__(self, eps: float = 0.1, ignore_index: int = -100) -> None:
        """Initialize.

        Args:
            eps: Smoothing weight in ``[0, 1]``.
            ignore_index: Label value to ignore.
        """
        super().__init__()
        if not (0.0 <= eps < 1.0):
            raise ValueError(f"eps must be in [0, 1), got {eps}")
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        V = shift_logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        nll = -log_probs.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        smooth = -log_probs.mean(dim=-1)
        pad_mask = shift_labels.eq(self.ignore_index)
        nll = nll.masked_fill(pad_mask, 0.0)
        smooth = smooth.masked_fill(pad_mask, 0.0)
        n_tokens = (~pad_mask).sum().clamp_min(1)
        loss = (1.0 - self.eps) * nll.sum() + self.eps * smooth.sum()
        return loss / n_tokens
