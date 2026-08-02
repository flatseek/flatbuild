"""DataModule — turns a list of tokenized samples into a fast DataLoader.

Tokens are pre-computed once at construction time; the dataset simply
indexes into the cached tensor lists at iteration time. This avoids
per-step tokenization, makes worker multiprocessing trivial (the
dataset is just two ``list[Tensor]`` fields), and keeps the entire
training set in RAM only for very compact models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class TokenizedTensorDataset(Dataset):
    """An in-memory dataset of pre-tokenized ``(input_ids, labels)`` pairs.

    Both fields are ``Tensor``s of dtype ``torch.long``. Padding is
    left to a downstream collate function so the dataset can also be
    iterated without padding (e.g. during evaluation when fixed shapes
    aren't required).
    """

    def __init__(self, rows: Sequence[tuple[list[int], list[int]]]) -> None:
        """Pre-tensorize every row.

        Args:
            rows: Iterable of ``(input_ids, labels)`` pairs. Each list
                becomes a ``torch.long`` tensor at construction time
                — so iteration is allocation-free.
        """
        ids_list: list[Tensor] = []
        labels_list: list[Tensor] = []
        for ids, labels in rows:
            if not ids:
                continue
            ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
        self._ids = ids_list
        self._labels = labels_list

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._ids[index], self._labels[index]


@dataclass
class DataLoaderConfig:
    """Knobs forwarded to :class:`torch.utils.data.DataLoader`."""

    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    drop_last: bool = True


def build_dataloader(
    tokenized_rows: Sequence[tuple[list[int], list[int]]],
    *,
    batch_size: int,
    shuffle: bool,
    pad_token_id: int,
    config: DataLoaderConfig | None = None,
) -> DataLoader:
    """Construct a :class:`DataLoader` over ``tokenized_rows``.

    Args:
        tokenized_rows: Pre-tokenized ``(input_ids, labels)`` rows.
        batch_size: Micro-batch size.
        shuffle: Whether to shuffle each epoch.
        pad_token_id: Padding token id; defaults to ``0`` if ``None``.
        config: Optional DataLoader overrides.

    Returns:
        A configured :class:`DataLoader`.
    """
    cfg = config or DataLoaderConfig()

    def _collate(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """Pad a list of variable-length tokenized rows into a batch.

        Building tensors from a Python list via ``torch.tensor`` with
        right-padding is faster than per-row ``torch.full`` calls.
        """
        if not batch:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        max_len = max(len(pair[0]) for pair in batch)
        pad = int(pad_token_id or 0)
        ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for row_idx, (ids_row, labels_row) in enumerate(batch):
            n = ids_row.numel()
            ids[row_idx, :n] = ids_row
            labels[row_idx, :n] = labels_row
        return ids, labels

    # Worker count > 0 requires sample-multiprocessing which fails for
    # in-process closures; the dataclass is already picklable.
    dataset = TokenizedTensorDataset(tokenized_rows)
    kwargs: dict = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
    )
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = cfg.persistent_workers
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(dataset, **kwargs)


__all__ = [
    "TokenizedTensorDataset",
    "DataLoaderConfig",
    "build_dataloader",
]
