"""Dataset loaders and sample normalization for Flatbuild."""

from flatbuild.datasets.base import (
    ConversationSample,
    InstructionSample,
    PretrainingSample,
    Sample,
    collate_conversation,
    collate_pretraining,
    normalize_sample,
)
from flatbuild.datasets.loader import (
    ParquetDataset,
    StreamingJSONLDataset,
    build_train_val_test,
    load_dataset,
)
from flatbuild.datasets.registry import DATASET_LOADERS, register_loader

__all__ = [
    # Sample normalization
    "PretrainingSample",
    "InstructionSample",
    "ConversationSample",
    "Sample",
    "normalize_sample",
    "collate_conversation",
    "collate_pretraining",
    # Dataset loaders
    "StreamingJSONLDataset",
    "ParquetDataset",
    "load_dataset",
    "build_train_val_test",
    "register_loader",
    "DATASET_LOADERS",
]
