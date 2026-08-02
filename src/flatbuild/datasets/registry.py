"""Format-name registry for dataset loaders."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from flatbuild.datasets.base import Sample

# Each entry maps a string format name (matching :class:`DatasetFormat`)
# to a callable ``(path, field_mapping) -> Iterator[dict]``.
DATASET_LOADERS: dict[str, Callable[..., Iterator[Any]]] = {}


def register_loader(name: str, loader: Callable[..., Iterator[Any]]) -> None:
    """Register a dataset loader.

    Args:
        name: Format name (``"jsonl"``, ``"parquet"``, …).
        loader: Callable returning an iterator of dict-like rows.
    """
    DATASET_LOADERS[name] = loader


def load_iter(path: Path, name: str, field_mapping: dict[str, str] | None = None) -> Iterator[dict]:
    """Iterate rows using a registered loader."""
    if name not in DATASET_LOADERS:
        raise KeyError(f"No loader registered for {name!r}")
    return DATASET_LOADERS[name](path=path, field_mapping=field_mapping or {})


__all__ = ["DATASET_LOADERS", "register_loader", "load_iter", "Sample"]
