"""Flatbuild - Train conversational language models from scratch.

Flatbuild is part of the Flat ecosystem. It exposes every component
of a small-scale Transformer training pipeline as an independent,
replaceable module, driven by a single YAML configuration.
"""

from flatbuild.utils import get_logger, setup_logging

__version__ = "0.1.1"
__author__ = "Flatbuild Contributors"

__all__ = [
    "__version__",
    "get_logger",
    "setup_logging",
]
