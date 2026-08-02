"""Trainer module for Flatbuild.

The training loop is split into:

- :mod:`flatbuild.trainer.datamodule` — pre-tokenized :class:`DataLoader`
  builder (caches tensors once; supports ``num_workers`` /
  ``persistent_workers`` / ``pin_memory``).
- :mod:`flatbuild.trainer.validation` — eval pass with
  :func:`torch.inference_mode` and an optional ``max_batches`` cap.
- :mod:`flatbuild.trainer.progress` — :class:`tqdm.auto.tqdm` per-epoch
  progress UI.
- :mod:`flatbuild.trainer.profiler` — phase timer with summary.
- :mod:`flatbuild.trainer.trainer` — orchestrator that wires it all
  together and exposes :class:`FlatbuildTrainer`.
"""

from flatbuild.trainer.datamodule import (
    DataLoaderConfig,
    TokenizedTensorDataset,
    build_dataloader,
)
from flatbuild.trainer.profiler import PerformanceProfiler
from flatbuild.trainer.progress import ProgressReporter
from flatbuild.trainer.tokenize import batch_samples, tokenize_sample
from flatbuild.trainer.trainer import FlatbuildTrainer, TrainArtifacts, build_callbacks
from flatbuild.trainer.validation import ValidationRunner, make_validation_runner

__all__ = [
    "FlatbuildTrainer",
    "TrainArtifacts",
    "build_callbacks",
    "ValidationRunner",
    "make_validation_runner",
    "TokenizedTensorDataset",
    "DataLoaderConfig",
    "build_dataloader",
    "ProgressReporter",
    "PerformanceProfiler",
    "tokenize_sample",
    "batch_samples",
]
