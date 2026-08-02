"""Filesystem helpers used across Flatbuild."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from flatbuild.utils.reproducibility import get_timestamp

_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(value: str) -> str:
    """Coerce ``value`` into a filename-safe slug.

    Args:
        value: Arbitrary string.

    Returns:
        Slug containing only ``A-Z``, ``a-z``, ``0-9``, ``-``, ``_``
        and ``.``. Empty input yields ``"untitled"``.
    """
    cleaned = _SAFE_PATTERN.sub("-", value.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "untitled"


def resolve_path(value: str | os.PathLike[str] | None) -> Path | None:
    """Expand user / environment variables in a path-like value.

    Args:
        value: Path-like value or ``None``.

    Returns:
        Expanded absolute :class:`pathlib.Path` or ``None``.
    """
    if value is None:
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _short_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp like ``20260801T153045Z``."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def create_run_folder(
    project: str | None = None,
    output_dir: str | Path = "outputs",
) -> Path:
    """Create a fresh ``<output_dir>/<project>/<timestamp>/`` folder.

    Args:
        project: Project slug (e.g. ``"demo"``). Defaults to ``"run"``.
        output_dir: Parent directory.

    Returns:
        Newly-created run folder.
    """
    slug = safe_filename(project or "run")
    root = Path(output_dir) / slug
    root.mkdir(parents=True, exist_ok=True)
    folder = root / _short_timestamp()
    counter = 1
    while folder.exists():
        folder = root / f"{_short_timestamp()}-{counter}"
        counter += 1
    folder.mkdir(parents=False)
    return folder


def get_run_folder(
    project: str | None = None,
    output_dir: str | Path = "outputs",
) -> Path:
    """Locate the most recent run folder for a project.

    Args:
        project: Project slug.
        output_dir: Parent directory to scan.

    Returns:
        The latest existing run folder.

    Raises:
        FileNotFoundError: If no run folder exists for the project.
    """
    slug = safe_filename(project or "run")
    root = Path(output_dir) / slug
    if not root.exists():
        raise FileNotFoundError(
            f"No runs found under {root}. Run `flatbuild train` first."
        )
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run folders under {root}.")
    return candidates[-1]


def write_json(path: Path, data: dict, indent: int = 2) -> None:
    """Write ``data`` to ``path`` as pretty-printed JSON.

    Args:
        path: Destination file path.
        data: JSON-serializable data.
        indent: Indentation level.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=False)
        f.write("\n")


def get_timestamp() -> str:
    """Re-export :func:`flatbuild.utils.reproducibility.get_timestamp`."""
    return get_timestamp()
