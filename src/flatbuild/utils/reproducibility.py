"""Reproducibility utilities for Flatbuild."""

from __future__ import annotations

import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def get_git_info() -> dict[str, Any]:
    """Collect git branch, commit and dirty flag.

    Returns:
        Dictionary with ``branch``, ``commit`` (short) and ``is_dirty``.
        Falls back to ``"unknown"`` when not inside a git repository.
    """
    info: dict[str, Any] = {
        "branch": "unknown",
        "commit": "unknown",
        "is_dirty": False,
    }
    try:
        for key, args in (
            ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
            ("commit", ["rev-parse", "HEAD"]),
            ("is_dirty", ["status", "--porcelain"]),
        ):
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                if key == "is_dirty":
                    info[key] = bool(result.stdout.strip())
                elif key == "commit":
                    info[key] = result.stdout.strip()[:8]
                else:
                    info[key] = result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return info


def get_timestamp() -> str:
    """Return the current time in ISO-8601 format, UTC."""
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_environment_info() -> dict[str, Any]:
    """Return a dictionary of platform + key package versions.

    Returns:
        Python version, platform, processor, and versions of common
        dependencies (``torch``, ``numpy``, ``tokenizers``,
        ``safetensors``, ``flatbuild``).
    """
    import platform

    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    for package in ("torch", "numpy", "tokenizers", "safetensors", "flatbuild"):
        try:
            module = __import__(package)
            info[package] = getattr(module, "__version__", "unknown")
        except (ImportError, NotImplementedError):
            info[package] = "not installed"
    return info


class ReproducibilityContext:
    """Context manager that seeds RNGs and collects reproducibility info."""

    def __init__(self, seed: int, include_env: bool = True) -> None:
        """Initialize the context.

        Args:
            seed: Seed value to apply on entry.
            include_env: When ``True``, environment info is included
                in :meth:`get_info`.
        """
        self.seed = seed
        self.include_env = include_env
        self._original_torch_seed: int | None = None

    def __enter__(self) -> "ReproducibilityContext":
        """Seed all RNGs and save the previous PyTorch seed (if any)."""
        try:
            import torch

            self._original_torch_seed = torch.initial_seed()
        except ImportError:
            pass
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Leave the context without re-seeding.

        Returns:
            ``False`` so exceptions propagate.
        """
        return False

    def get_info(self) -> dict[str, Any]:
        """Return reproducibility metadata.

        Returns:
            Dictionary with ``seed``, ``git``, ``timestamp`` and
            optionally ``environment`` info.
        """
        info: dict[str, Any] = {
            "seed": self.seed,
            "git": get_git_info(),
            "timestamp": get_timestamp(),
        }
        if self.include_env:
            info["environment"] = get_environment_info()
        return info
