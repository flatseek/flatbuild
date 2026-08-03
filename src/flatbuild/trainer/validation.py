"""ValidationRunner — a small wrapper around the eval pass.

Goals:
- model.eval() + torch.inference_mode() (no autograd graph)
- batching with optional ``max_batches`` cap
- no shuffle
- pin_memory only when CUDA is available
- compute loss / perplexity / token accuracy once per batch
- a single ``summarize()`` to merge metrics across batches
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from flatbuild.metrics import LanguageModelingMetrics
from flatbuild.models import FlatbuildModel
from flatbuild.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationConfig:
    """Runtime configuration for :class:`ValidationRunner`."""

    max_batches: int | None = None  # if set, stop after this many batches
    device: torch.device | str | None = None  # inference device


class ValidationRunner:
    """Run the validation pass and summarize metrics.

    Args:
        model: The trained model (will be set to ``eval()`` mode).
        tokenizer: Trained tokenizer (used to determine ``pad_token_id``
            — defaults to ``0`` if absent).
        loader: DataLoader yielding ``(input_ids, labels)`` tuples.
            Shuffle should already be off.
        device: Where to run forward. ``None`` keeps the model on its
            current device.
        max_batches: Cap on the number of batches evaluated per call.
            ``None`` evaluates every batch in the loader.
    """

    def __init__(
        self,
        model: FlatbuildModel,
        tokenizer,
        loader: DataLoader,
        *,
        max_batches: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the runner."""
        self.model = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.max_batches = max_batches
        self.device = torch.device(device) if device is not None else None

    def run(self) -> dict[str, float]:
        """Run validation once and return ``{loss, perplexity, accuracy}``.

        Uses ``torch.inference_mode()`` — no autograd graph is built,
        no gradient checkpointing runs, and tensors stay on device
        without cloning. The model is left in ``eval()`` mode.

        Returns:
            Dictionary of scalar metrics. Empty ``loader`` yields
            a degenerate ``{loss: 0, perplexity: 1, accuracy: 0}``.
        """
        was_training = self.model.training
        target_device = self.device or next(self.model.parameters()).device

        if len(self.loader) == 0:
            return {"loss": 0.0, "perplexity": 1.0, "accuracy": 0.0}

        try:
            self.model.eval()
            accum = LanguageModelingMetrics.empty()
            n_batches = 0
            with torch.inference_mode():
                for batch_idx, (input_ids, labels) in enumerate(self.loader):
                    if self.max_batches is not None and batch_idx >= self.max_batches:
                        break
                    input_ids = input_ids.to(target_device, non_blocking=True)
                    labels = labels.to(target_device, non_blocking=True)
                    out = self.model(input_ids, labels=labels)
                    accum.update(out.loss, out.logits, labels, ignore_index=-100)
                    n_batches += 1
            return accum.summary()
        finally:
            if was_training:
                self.model.train()


# Convenience wrapper used by FlatbuildTrainer


def make_validation_runner(
    model: FlatbuildModel,
    tokenizer,
    tokenized_rows,
    *,
    batch_size: int,
    max_batches: int | None,
    pad_token_id: int,
    pin_memory: bool = False,
    device: torch.device | str | None = None,
) -> ValidationRunner:
    """Build a fresh :class:`ValidationRunner` from raw tokenized rows.

    Args:
        model: Trained model.
        tokenizer: Tokenizer (for ``pad_token_id``).
        tokenized_rows: Iterable of ``(input_ids, labels)`` pairs.
        batch_size: Batch size (typically ``= trainer.batch_size``).
        max_batches: Optional cap; otherwise evaluates everything.
        pad_token_id: Padding token id; defaults to ``0`` if ``None``.
        pin_memory: Whether to use CUDA pinned memory.
        device: Optional inference device.

    Returns:
        Configured :class:`ValidationRunner` ready to ``.run()``.
    """
    from flatbuild.trainer.datamodule import DataLoaderConfig, build_dataloader

    cfg = DataLoaderConfig(
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=False,  # evaluate the final partial batch
    )
    loader = build_dataloader(
        tokenized_rows,
        batch_size=batch_size,
        shuffle=False,
        pad_token_id=pad_token_id,
        config=cfg,
    )
    return ValidationRunner(
        model,
        tokenizer,
        loader,
        max_batches=max_batches,
        device=device,
    )


__all__ = [
    "ValidationConfig",
    "ValidationRunner",
    "make_validation_runner",
]
