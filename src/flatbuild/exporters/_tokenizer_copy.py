"""Helpers for copying a tokenizer directory alongside an exported model.

Flatbuild stores BPE tokenizer artifacts in a directory like:

    outputs/<run>/tokenizer/
        tokenizer.json
        tokenizer_config.json
        chat_template.json

The HuggingFace / llama.cpp / flatrun convention is to put those
files at the *same level* as the exported weights, e.g.::

    exported/
        config.json
        model.safetensors
        tokenizer.json
        tokenizer_config.json

So ``shutil.copytree(tok, out/tokenizer)`` (which puts everything
under a subfolder) is wrong. This module provides a deterministic,
flat copy that places every tokenizer file directly under the export
root.

If a tokenizer directory is missing entirely, :func:`copy_tokenizer`
is a no-op (warnings logged) — exporting without a tokenizer is
allowed for weight-only flows (e.g. partial fine-tune of a tokenizer
artifact already in another path).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from flatbuild.utils import get_logger

logger = get_logger(__name__)


# Files we always skip when laying tokenizer files flat — these are
# not part of the canonical tokenizer artifact (model weights might
# still mention them in the manifest but they aren't safe to ship).
_NON_TOKENIZER_FILES = {"manifest.json"}

# HF expects the chat template as ``tokenizer_config.json`` and may also
# accept ``chat_template.json`` — keep both flat.
_TOKENIZER_FILE_SUFFIXES = (".json", ".txt", ".model", ".tiktoken")


def _looks_like_tokenizer_file(path: Path) -> bool:
    """Return ``True`` if ``path`` looks like a tokenizer artifact file."""
    if not path.is_file():
        return False
    if path.name in _NON_TOKENIZER_FILES:
        return False
    if not path.suffix:
        return False
    return path.suffix.lower() in _TOKENIZER_FILE_SUFFIXES


def copy_tokenizer(
    tokenizer_dir: str | Path | None,
    output_dir: str | Path,
) -> int:
    """Copy tokenizer artifacts from ``tokenizer_dir`` flat into ``output_dir``.

    Args:
        tokenizer_dir: Source directory containing the BPE tokenizer
            files. ``None`` is treated as a no-op.
        output_dir: Destination directory where ``tokenizer.json``
            etc. will be placed directly (no nesting).

    Returns:
        Number of files actually copied.
    """
    if tokenizer_dir is None:
        return 0
    src = Path(tokenizer_dir)
    if not src.is_dir():
        logger.warning(f"Tokenizer directory not found: {src} — skipping tokenizer copy.")
        return 0

    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for src_file in src.iterdir():
        if not _looks_like_tokenizer_file(src_file):
            continue
        target = dst / src_file.name
        if target.exists():
            target.unlink()
        try:
            shutil.copy2(src_file, target)
        except OSError as exc:  # pragma: no cover - best-effort copy
            logger.warning(f"Could not copy {src_file.name}: {exc}")
            continue
        count += 1

    logger.info(f"Copied {count} tokenizer file(s) flat into {dst}")
    return count


__all__ = ["copy_tokenizer"]
