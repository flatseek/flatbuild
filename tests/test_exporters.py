"""Tests for the checkpoint exporters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flatbuild.config import (
    ChatTemplateConfig,
    DatasetConfig,
    ExportConfig,
    FlatBuildConfig,
    GenerateConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainerConfig,
    CheckpointConfig,
)
from flatbuild.exporters.huggingface import HuggingFaceExporter
from flatbuild.exporters.safetensors import SafeTensorsExporter
from flatbuild.models import FlatbuildModel
from flatbuild.tokenizers.bpe import BPETokenizer


def _tiny_config() -> FlatBuildConfig:
    return FlatBuildConfig(
        name="test",
        dataset=DatasetConfig(),
        tokenizer=TokenizerConfig(),
        chat_template=ChatTemplateConfig(),
        model=ModelConfig(
            vocab_size=64,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            hidden_dim=32,
            ffn_dim=128,
            context_length=64,
        ),
        optimizer=OptimizerConfig(),
        scheduler=SchedulerConfig(),
        trainer=TrainerConfig(),
        checkpoint=CheckpointConfig(),
        export=ExportConfig(),
        generate=GenerateConfig(),
    )


def _fixture_tokenizer(tmp_path: Path) -> Path:
    """Train a tiny BPE tokenizer and return its directory."""
    tok_dir = tmp_path / "tok"
    tok = BPETokenizer.train(
        ["hello world", "goodbye world"], vocab_size=32, min_frequency=1
    )
    tok.save(tok_dir)
    return tok_dir


def test_safetensors_export(tmp_path):
    """SafeTensors exporter writes model.safetensors + Llama-compatible config.json."""
    cfg = _tiny_config()
    model = FlatbuildModel(cfg.model)
    exporter = SafeTensorsExporter(copy_tokenizer=False)
    out = exporter.export(model, tmp_path / "export")
    assert (out / "model.safetensors").exists()
    cfg_json = json.loads((out / "config.json").read_text())
    assert cfg_json["hidden_size"] == cfg.model.hidden_dim
    assert cfg_json["num_attention_heads"] == cfg.model.n_heads


def test_safetensors_tokenizer_attached_flat(tmp_path):
    """Tokenizer files land at root (next to model.safetensors), not under ``tokenizer/``."""
    cfg = _tiny_config()
    model = FlatbuildModel(cfg.model)
    tok_dir = _fixture_tokenizer(tmp_path)
    exporter = SafeTensorsExporter(copy_tokenizer=True)
    out = exporter.export(model, tmp_path / "out", tokenizer_dir=tok_dir)

    # Weight side.
    assert (out / "model.safetensors").exists()
    assert (out / "config.json").exists()
    # Tokenizer side — flat.
    assert (out / "tokenizer.json").exists()
    assert (out / "tokenizer_config.json").exists()
    # And explicitly NOT under a subfolder.
    assert not (out / "tokenizer").exists()


def test_huggingface_export(tmp_path):
    """HuggingFace exporter writes model_type=llama with the expected config keys."""
    cfg = _tiny_config()
    model = FlatbuildModel(cfg.model)
    exporter = HuggingFaceExporter(copy_tokenizer=False)
    out = exporter.export(model, tmp_path / "export_hf")
    cfg_json = json.loads((out / "config.json").read_text())
    assert cfg_json["model_type"] == "llama"
    assert "generation_config.json" in {p.name for p in out.iterdir()}


def test_huggingface_tokenizer_attached_flat(tmp_path):
    """HuggingFace exporter drops tokenizer files at root, not under ``tokenizer/``."""
    cfg = _tiny_config()
    model = FlatbuildModel(cfg.model)
    tok_dir = _fixture_tokenizer(tmp_path)
    exporter = HuggingFaceExporter(copy_tokenizer=True)
    out = exporter.export(model, tmp_path / "out_hf", tokenizer_dir=tok_dir)

    assert (out / "model.safetensors").exists()
    assert (out / "tokenizer.json").exists()
    assert (out / "tokenizer_config.json").exists()
    assert not (out / "tokenizer").exists()

