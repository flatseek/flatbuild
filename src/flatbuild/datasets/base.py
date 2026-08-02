"""Sample types used inside Flatbuild.

A dataset on disk can be JSONL, Parquet or a HuggingFace dataset.
Regardless of the source, every sample in memory is normalized
into one of three shapes:

- :class:`PretrainingSample` — a single piece of plain text.
- :class:`InstructionSample` — instruction/input/output triple.
- :class:`ConversationSample` — list of role/content messages.

Keeping these shapes small and immutable makes the rest of the
pipeline (tokenization, batching, loss masking) easy to reason about.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(frozen=True)
class PretrainingSample:
    """A single piece of unsupervised text."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InstructionSample:
    """A supervised instruction triple."""

    instruction: str
    output: str
    input: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationSample:
    """A multi-turn message list.

    Roles are typically ``"system"``, ``"user"``, ``"assistant"``.
    """

    messages: tuple[tuple[str, str], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("ConversationSample requires at least one message")
        for role, content in self.messages:
            if not isinstance(role, str) or not isinstance(content, str):
                raise TypeError(
                    f"ConversationSample expects (str, str) tuples, got {(type(role).__name__, type(content).__name__)}"
                )


Sample = Union[PretrainingSample, InstructionSample, ConversationSample]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_sample(raw: dict[str, Any]) -> Sample:
    """Convert a raw dictionary into the correct sample type.

    Heuristics:
        * If the dict has a ``messages`` key with list-of-dict entries
          → :class:`ConversationSample`.
        * If the dict has ``instruction`` (and ``output`` or
          ``response``) → :class:`InstructionSample`.
        * Otherwise, if the dict has ``text`` → :class:`PretrainingSample`.
        * If the dict has ``input`` → :class:`InstructionSample` with that
          as the optional input field.

    Args:
        raw: Dictionary loaded from a dataset row.

    Returns:
        A normalized :data:`Sample`.

    Raises:
        ValueError: If ``raw`` carries no recognizable fields.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"normalize_sample expects a dict, got {type(raw).__name__}")

    metadata = raw.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {"_original_metadata": metadata}

    if "messages" in raw:
        messages_raw = raw["messages"]
        if not isinstance(messages_raw, Iterable):
            raise TypeError("`messages` must be an iterable of role/content pairs")
        messages: list[tuple[str, str]] = []
        for entry in messages_raw:
            if not isinstance(entry, dict):
                raise TypeError(f"Message entries must be dicts, got {type(entry).__name__}")
            role = entry.get("role")
            content = entry.get("content")
            if role is None or content is None:
                raise ValueError(f"Message needs 'role' and 'content', got {entry!r}")
            messages.append((str(role), str(content)))
        return ConversationSample(messages=tuple(messages), metadata=dict(metadata))

    if "instruction" in raw:
        return InstructionSample(
            instruction=str(raw["instruction"]),
            output=str(raw.get("output", raw.get("response", ""))),
            input=(None if raw.get("input") in (None, "") else str(raw["input"])),
            metadata=dict(metadata),
        )

    if "text" in raw and raw["text"]:
        return PretrainingSample(text=str(raw["text"]), metadata=dict(metadata))

    raise ValueError(
        "Sample must contain one of: 'messages', 'instruction', or 'text'. "
        f"Got keys: {sorted(raw.keys())}"
    )


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------


def collate_pretraining(
    samples: Iterable[PretrainingSample],
) -> list[PretrainingSample]:
    """Collect pretraining samples into a list (no padding yet)."""
    return list(samples)


def collate_conversation(
    samples: Iterable[ConversationSample],
) -> list[ConversationSample]:
    """Collect conversation samples into a list."""
    return list(samples)
