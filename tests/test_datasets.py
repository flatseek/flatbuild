"""Tests for dataset normalization and the JSONL loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flatbuild.datasets.base import (
    ConversationSample,
    InstructionSample,
    PretrainingSample,
    normalize_sample,
)
from flatbuild.datasets.loader import (
    DatasetSplit,
    StreamingJSONLDataset,
    build_train_val_test,
)


def test_normalize_conversation():
    """Conversation dicts with role/content entries become ConversationSample."""
    raw = {
        "messages": [
            {"role": "system", "content": "You are Flatbot."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    sample = normalize_sample(raw)
    assert isinstance(sample, ConversationSample)
    assert sample.messages == (
        ("system", "You are Flatbot."),
        ("user", "Hi"),
        ("assistant", "Hi there!"),
    )


def test_normalize_instruction():
    """Instruction dicts are mapped to InstructionSample."""
    raw = {
        "instruction": "Translate.",
        "input": "hello",
        "output": "bonjour",
    }
    sample = normalize_sample(raw)
    assert isinstance(sample, InstructionSample)
    assert sample.instruction == "Translate."
    assert sample.input == "hello"
    assert sample.output == "bonjour"


def test_normalize_pretraining_fallback():
    """Plain text dicts become PretrainingSample."""
    raw = {"text": "Once upon a time."}
    sample = normalize_sample(raw)
    assert isinstance(sample, PretrainingSample)
    assert sample.text == "Once upon a time."


def test_normalize_invalid_raises():
    """Samples without recognizable fields raise ValueError."""
    with pytest.raises(ValueError):
        normalize_sample({"foo": "bar"})


def test_jsonl_loader(tmp_path: Path):
    """StreamingJSONLDataset yields one Sample per non-empty line."""
    path = tmp_path / "data.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
        {"text": "hello"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    samples = list(StreamingJSONLDataset(path))
    assert len(samples) == 2
    # Note: the loader yields raw dicts, not Samples; this is a regression
    # anchor so we notice if the contract changes.
    assert isinstance(samples[0], dict)


def test_field_mapping(tmp_path: Path):
    """``field_mapping`` rewrites key names on the fly."""
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"prompt": "Q?", "response": "A!"}) + "\n")

    ds = StreamingJSONLDataset(path, field_mapping={"instruction": "prompt", "output": "response"})
    rows = list(ds)
    assert rows[0]["instruction"] == "Q?"
    assert rows[0]["output"] == "A!"


def test_split_deterministic():
    """Same seed => same partition."""
    samples = [PretrainingSample(text=str(i)) for i in range(100)]
    a = build_train_val_test(samples, train_split=0.7, val_split=0.2, test_split=0.1, seed=42)
    b = build_train_val_test(samples, train_split=0.7, val_split=0.2, test_split=0.1, seed=42)
    assert [s.text for s in a.train] == [s.text for s in b.train]
    assert isinstance(a, DatasetSplit)


def test_split_different_seeds_differ():
    """Different seeds differ (probabilistic check)."""
    samples = [PretrainingSample(text=str(i)) for i in range(100)]
    a = build_train_val_test(samples, train_split=0.7, val_split=0.2, test_split=0.1, seed=42)
    b = build_train_val_test(samples, train_split=0.7, val_split=0.2, test_split=0.1, seed=99)
    assert [s.text for s in a.train] != [s.text for s in b.train]
