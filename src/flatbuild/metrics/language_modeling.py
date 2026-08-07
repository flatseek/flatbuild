"""Language-modeling metrics: perplexity and token accuracy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


def compute_perplexity(losses: Iterable[float]) -> float:
    """Compute perplexity from an iterable of average per-token losses.

    Args:
        losses: Iterable of scalar losses (one per evaluation micro-batch).

    Returns:
        Perplexity (``exp(mean_loss)``) clamped to ``[1.0, 1e9]``.
    """
    values = [float(l) for l in losses]
    if not values:
        return 1.0
    avg = sum(values) / len(values)
    return min(1e9, max(1.0, math.exp(avg)))


def compute_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute next-token accuracy, ignoring ``ignore_index`` positions.

    Args:
        logits: Tensor of shape ``(B, T, V)``.
        labels: Tensor of shape ``(B, T)``.
        ignore_index: Label value to ignore.

    Returns:
        Accuracy as a float in ``[0.0, 1.0]``.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    preds = shift_logits.argmax(dim=-1)
    mask = shift_labels.ne(ignore_index)
    correct = (preds.eq(shift_labels) & mask).sum().item()
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return correct / total


@dataclass
class LanguageModelingMetrics:
    """Aggregating container that tracks loss / accuracy / perplexity."""

    losses: list[float]
    correct: int
    total: int

    @classmethod
    def empty(cls) -> "LanguageModelingMetrics":
        """Return a fresh accumulator."""
        return cls(losses=[], correct=0, total=0)

    def update(self, loss: float, logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> None:
        """Accumulate a single micro-batch of predictions."""
        self.losses.append(float(loss.detach().cpu().item()) if torch.is_tensor(loss) else float(loss))
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        preds = shift_logits.argmax(dim=-1)
        mask = shift_labels.ne(ignore_index)
        self.correct += int((preds.eq(shift_labels) & mask).sum().item())
        self.total += int(mask.sum().item())

    def summary(self) -> dict[str, float]:
        """Return ``{loss, perplexity, accuracy}`` as plain floats."""
        valid_losses = [l for l in self.losses if l == l]  # filter NaN
        if not valid_losses:
            return {
                "loss": float("nan"),
                "perplexity": 1.0,
                "accuracy": 0.0,
            }
        avg_loss = sum(valid_losses) / len(valid_losses)
        return {
            "loss": avg_loss,
            "perplexity": compute_perplexity(valid_losses),
            "accuracy": self.correct / self.total if self.total else 0.0,
        }
