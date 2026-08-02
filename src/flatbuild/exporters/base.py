"""Base exporter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from flatbuild.config import ExportConfig
from flatbuild.models import FlatbuildModel


class Exporter(ABC):
    """Abstract base for checkpoint exporters."""

    @abstractmethod
    def export(
        self,
        model: FlatbuildModel,
        output_dir: str | Path,
        *,
        config: ExportConfig | None = None,
    ) -> Path:
        """Export the model to ``output_dir``.

        Args:
            model: Trained FlatbuildModel.
            output_dir: Destination directory.
            config: Optional per-exporter config.

        Returns:
            The directory the model was exported to.
        """
