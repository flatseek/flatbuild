"""Dataset loaders.

Supported sources:
- ``jsonl`` — line-delimited JSON.
- ``parquet`` — columnar file, read into memory.
- ``hf`` — HuggingFace ``datasets.load_dataset`` (streaming or cached).

Every loader yields normalized :class:`Sample` objects via
:func:`flatbuild.datasets.base.normalize_sample`.

Splits are deterministic given a seed.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flatbuild.config import DatasetConfig, DatasetFormat
from flatbuild.datasets.base import Sample, normalize_sample
from flatbuild.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


@dataclass
class StreamingJSONLDataset:
    """Yield rows from a JSONL file one at a time."""

    path: Path
    field_mapping: dict[str, str] | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL dataset not found: {self.path}")

    def __iter__(self) -> Iterator[Sample]:
        with open(self.path, encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON at {self.path}:{line_no}: {exc}"
                    ) from exc
                yield self._apply_field_mapping(obj)

    def _apply_field_mapping(self, obj: dict) -> dict:
        """Return a new dict with keys renamed per ``field_mapping``."""
        if not self.field_mapping:
            return obj
        new = dict(obj)
        for target, source in self.field_mapping.items():
            if source in obj:
                new[target] = obj[source]
        return new


@dataclass
class ParquetDataset:
    """Read a Parquet file (or its rows) into memory."""

    path: Path
    field_mapping: dict[str, str] | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Parquet dataset not found: {self.path}")

    def __iter__(self) -> Iterator[Sample]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Parquet support requires pyarrow. `pip install pyarrow`"
            ) from exc

        table = pq.read_table(self.path)
        columns = table.column_names
        n = table.num_rows
        mapping = self.field_mapping or {}
        for i in range(n):
            row: dict[str, Any] = {col: table.column(col)[i].as_py() for col in columns}
            for target, source in mapping.items():
                if source in row:
                    row[target] = row[source]
            try:
                yield normalize_sample(row)
            except ValueError:
                # Skip rows that don't carry usable content.
                continue
        # Silence the unused-variable warning.
        _ = n


# ---------------------------------------------------------------------------
# Hub loader
# ---------------------------------------------------------------------------


def load_dataset(
    config: DatasetConfig,
    *,
    base_dir: Path | None = None,
) -> Iterator[Sample]:
    """Yield samples based on :class:`DatasetConfig`.

    Args:
        config: Dataset configuration.
        base_dir: Optional base directory; ``config.path`` is resolved
            relative to this if not already absolute.

    Returns:
        An iterator over normalized samples.
    """
    path = Path(config.path)
    if base_dir is not None and not path.is_absolute():
        path = (base_dir / path).resolve()

    logger.info(
        f"Loading dataset: format={config.format.value}, path={path}, "
        f"max_samples={config.max_samples}"
    )

    if config.format == DatasetFormat.JSONL:
        ds: Iterator[dict] = (
            obj for obj in _iter_jsonl(path, config.field_mapping)
        )
        samples = (normalize_sample(obj) for obj in ds)
    elif config.format == DatasetFormat.PARQUET:
        samples = iter(ParquetDataset(path=path, field_mapping=config.field_mapping))
    elif config.format == DatasetFormat.HF:
        samples = iter(_iter_hf(path, config))
    else:
        raise ValueError(f"Unsupported dataset format: {config.format}")

    if config.max_samples is not None:
        samples = _take_n(samples, config.max_samples)
    return samples


def _iter_jsonl(path: Path, field_mapping: dict[str, str] | None) -> Iterator[dict]:
    ds = StreamingJSONLDataset(path=path, field_mapping=field_mapping or {})
    for sample in ds:
        # Convert dataclass back to dict so normalize_sample can run.
        if hasattr(sample, "__dict__"):
            yield sample.__dict__
        else:
            yield dict(sample)  # type: ignore[arg-type]


def _iter_hf(path: Path, config: DatasetConfig) -> Iterator[Sample]:
    """Stream rows from a HuggingFace dataset name or directory."""
    try:
        from datasets import load_dataset as _hf_load
    except ImportError as exc:
        raise ImportError(
            "HF dataset support requires `pip install datasets`"
        ) from exc

    name = str(path)
    kwargs = dict(config.hf_kwargs or {})
    builder = _hf_load(name, **kwargs)
    for split_name in builder.keys():
        for row in builder[split_name]:
            try:
                yield normalize_sample(dict(row))
            except ValueError:
                continue
        # Only consume the first split unless the user is more specific.
        return


def _take_n(it: Iterator[Sample], n: int) -> Iterator[Sample]:
    """Yield at most ``n`` items from an iterator."""
    for i, item in enumerate(it):
        if i >= n:
            return
        yield item


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@dataclass
class DatasetSplit:
    """A materialised train/val/test split."""

    train: list[Sample]
    val: list[Sample]
    test: list[Sample]

    def sizes(self) -> dict[str, int]:
        """Return split sizes as a plain dictionary."""
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def build_train_val_test(
    samples: list[Sample],
    *,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> DatasetSplit:
    """Deterministically split a list of samples into train / val / test.

    Args:
        samples: All samples.
        train_split, val_split, test_split: Fractions, summed must be ``<=1``.
        seed: RNG seed.

    Returns:
        :class:`DatasetSplit` with materialized lists.

    Raises:
        ValueError: If split ratios are invalid.
    """
    total = train_split + val_split + test_split
    if total <= 0 or total > 1.0 + 1e-6:
        raise ValueError(
            f"train_split + val_split + test_split must be in (0, 1], got {total}"
        )

    order = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(order)

    n = len(samples)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    test = [samples[i] for i in test_idx]
    return DatasetSplit(train=train, val=val, test=test)
