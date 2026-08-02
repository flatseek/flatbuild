"""Tests for the training loop and checkpoint round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flatbuild.callbacks.base import Callback, CallbackContext
from flatbuild.checkpoint.manager import CheckpointManager, CheckpointState
from flatbuild.config import (
    DatasetConfig,
    FlatBuildConfig,
    ModelConfig,
    OptimizerConfig,
    Precision,
    SchedulerConfig,
    TokenizerConfig,
    TrainerConfig,
    CheckpointConfig,
    ExportConfig,
    GenerateConfig,
    ChatTemplateConfig,
)
from flatbuild.datasets.base import ConversationSample, normalize_sample
from flatbuild.models import FlatbuildModel
from flatbuild.tokenizers.bpe import BPETokenizer
from flatbuild.tokenizers.template import build_chat_template
from flatbuild.trainer.tokenize import batch_samples, tokenize_sample


def _demo_corpus():
    """Tiny but consistent conversation corpus for trainer tests."""
    return [
        ConversationSample(messages=(
            ("user", "Hi"),
            ("assistant", "Hello!"),
        )),
        ConversationSample(messages=(
            ("user", "What is 1+1?"),
            ("assistant", "Two."),
        )),
        ConversationSample(messages=(
            ("user", "Translate hello to Spanish."),
            ("assistant", "'Hello' is 'hola' in Spanish."),
        )),
        ConversationSample(messages=(
            ("user", "Goodbye"),
            ("assistant", "See you!"),
        )),
    ] * 6


def _make_config(vocab_size: int = 64) -> FlatBuildConfig:
    """Tiny model config for tests."""
    return FlatBuildConfig(
        name="test",
        dataset=DatasetConfig(path=""),
        tokenizer=TokenizerConfig(vocab_size=vocab_size),
        chat_template=ChatTemplateConfig(),
        model=ModelConfig(
            vocab_size=vocab_size,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            hidden_dim=32,
            ffn_dim=128,
            context_length=64,
        ),
        optimizer=OptimizerConfig(lr=1e-3),
        scheduler=SchedulerConfig(warmup_steps=1),
        trainer=TrainerConfig(
            epochs=1,
            batch_size=2,
            gradient_accumulation=2,
            max_steps=3,
            precision=Precision.FP32,
            eval_every_n_steps=None,
            log_every_n_steps=1,
            max_grad_norm=1.0,
        ),
        checkpoint=CheckpointConfig(every_n_steps=10, keep_last=1, save_final=True),
        export=ExportConfig(),
        generate=GenerateConfig(),
    )


def test_tokenize_and_batch_round_trip():
    """tokenize_sample plus batch_samples yields well-shaped tensors."""
    corpus = _demo_corpus()
    tok = BPETokenizer.train(["hello", "translate", "goodbye", "Two"], vocab_size=64, min_frequency=1)
    tmpl = build_chat_template(ChatTemplateConfig())
    rows = [tokenize_sample(s, tok, tmpl, max_length=64) for s in corpus]
    input_ids, labels = batch_samples(rows, pad_token_id=tok.pad_token_id or 0)
    assert input_ids.shape == labels.shape
    assert (labels >= -100).all()


def test_train_loop_saves_checkpoint(tmp_path):
    """A short training run saves a usable checkpoint."""
    samples = _demo_corpus()
    tok = BPETokenizer.train(
        [s.text if hasattr(s, "text") else " ".join(c for _, c in s.messages) for s in samples],
        vocab_size=64,
        min_frequency=1,
    )
    cfg = _make_config(vocab_size=tok.vocab_size)
    cfg.trainer.epochs = 1
    cfg.trainer.max_steps = 2
    cfg.trainer.gradient_accumulation = 1
    run_dir = tmp_path / "run"
    from flatbuild.trainer.trainer import build_callbacks, FlatbuildTrainer

    model = FlatbuildModel(cfg.model)
    trainer = FlatbuildTrainer(
        config=cfg,
        model=model,
        tokenizer=tok,
        samples=samples[:8],
        val_samples=samples[8:],
        run_dir=run_dir,
        callbacks=build_callbacks(cfg, run_dir),
    )
    artifacts = trainer.train()
    assert artifacts.run_dir.exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "checkpoints" / "final" / "model.safetensors").exists()


def test_checkpoint_round_trip(tmp_path):
    """Saved checkpoint reloads into a fresh model with matching weights."""
    cfg = _make_config(vocab_size=64)
    model = FlatbuildModel(cfg.model)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    mgr = CheckpointManager(run_dir, max_to_keep=2)
    state = CheckpointState(global_step=10, epoch_index=1, last_loss=2.5)
    out = mgr.save_final(
        model=model,
        optimizer=None,
        tokenizer_dir=None,
        config=cfg,
        state=state,
    )
    assert out.exists()

    bundle = CheckpointManager.load(out)
    assert bundle["config"].trainer.max_steps == cfg.trainer.max_steps
    assert bundle["state"].global_step == 10

    # Rebuild and reload.
    cfg2 = _make_config(vocab_size=64)
    model2 = FlatbuildModel(cfg2.model)
    model2.load_state_dict_llama(bundle["model_state_dict"], strict=False)
    # Cross-check that a couple of key tensors match.
    sd_a = model.state_dict_llama()
    sd_b = model2.state_dict_llama()
    for key in ("model.embed_tokens.weight", "model.norm.weight"):
        assert torch.allclose(sd_a[key], sd_b[key], atol=1e-6)
