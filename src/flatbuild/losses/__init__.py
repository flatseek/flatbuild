"""Loss functions for Flatbuild."""

from flatbuild.losses.cross_entropy import CrossEntropyLoss, LabelSmoothingLoss

__all__ = ["CrossEntropyLoss", "LabelSmoothingLoss"]
