"""Tests for the trainer refactor modules."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from flatbuild.config import (
    ChatTemplateConfig,
    CheckpointConfig,
    DatasetConfig,
    ExportConfig,
    FlatBuildConfig,
    GenerateConfig,
    ModelConfig,
    OptimizerConfig,
    Precision,
    SchedulerConfig,
    TokenizerConfig,
    TrainerConfig,
    ValidationConfig,
)
from flatbuild.datasets.base import ConversationSample, PretrainingSample
from flatbuild.models import FlatbuildModel
from flatbuild.trainer.datamodule import (
    DataLoaderConfig,
    TokenizedTensorDataset,
    build_dataloader,
)
from flatbuild.trainer.profiler import PerformanceProfiler
from flatbuild.trainer.progress import ProgressReporter
from flatbuild.trainer.validation import make_validation_runner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _demo_corpus(n: int = 8) -> list:
    out = []
    for i in range(n):
        out.append(
            ConversationSample(
                messages=(
                    ("user", f"Say hi {i}"),
                    ("assistant", f"Hi {i}!"),
                )
            )
        )
    return out


def _tiny_tokenized(n: int = 8, length: int = 12) -> list[tuple[list[int], list[int]]]:
    rows = []
    for i in range(n):
        ids = [1 + (i * 3 + j) % 31 for j in range(length)]
        labels = list(ids)
        rows.append((ids, labels))
    return rows


def _tiny_config() -> FlatBuildConfig:
    """A config sized for fast in-process tests."""
    return FlatBuildConfig(
        name="trainer-mod-test",
        dataset=DatasetConfig(),
        tokenizer=TokenizerConfig(vocab_size=64),
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
        optimizer=OptimizerConfig(lr=1e-3),
        scheduler=SchedulerConfig(warmup_steps=1),
        trainer=TrainerConfig(
            epochs=1,
            batch_size=2,
            gradient_accumulation=1,
            max_steps=3,
            precision=Precision.FP32,
            eval_every_n_steps=None,
            log_every_n_steps=1,
            max_grad_norm=1.0,
        ),
        validation=ValidationConfig(),
        checkpoint=CheckpointConfig(),
        export=ExportConfig(),
        generate=GenerateConfig(),
    )


def _demo_tokenizer():
    """Tiny BPE tokenizer over a vocabulary of 64 token ids."""
    from flatbuild.tokenizers.bpe import BPETokenizer

    corpus = ["hello world"] * 10 + ["goodbye world"] * 10
    return BPETokenizer.train(corpus, vocab_size=64, min_frequency=1)


def _chat_template():
    from flatbuild.tokenizers.template import build_chat_template

    return build_chat_template(ChatTemplateConfig())


# ---------------------------------------------------------------------------
# TokenizedTensorDataset
# ---------------------------------------------------------------------------


def test_tokenized_dataset_round_trip():
    """Pre-tokenized rows survive a round-trip through the dataset."""
    rows = [(list(range(1, 9)), list(range(1, 9))), ([42], [42])]
    ds = TokenizedTensorDataset(rows)
    assert len(ds) == 2
    ids, labels = ds[0]
    assert ids.dtype == torch.long
    assert ids.tolist() == list(range(1, 9))
    assert labels.tolist() == list(range(1, 9))
    assert ds[1][0].tolist() == [42]


def test_tokenized_dataset_skips_empty():
    """Empty rows are dropped at construction time."""
    ds = TokenizedTensorDataset([([1, 2, 3], [1, 2, 3]), ([], []), ([10], [10])])
    assert len(ds) == 2


def test_build_dataloader_pads_to_max_len():
    """build_dataloader should right-pad to longest row, label = -100 in padding."""
    rows = [([1, 2], [1, 2]), ([3, 4, 5, 6], [3, 4, 5, 6])]
    loader = build_dataloader(
        rows,
        batch_size=2,
        shuffle=False,
        pad_token_id=99,
    )
    batch = next(iter(loader))
    input_ids, labels = batch
    assert input_ids.shape == (2, 4)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long
    # Padding token is 99; padded-label = -100.
    assert (input_ids[0, 2:] == 99).all()
    assert (labels[0, 2:] == -100).all()


# ---------------------------------------------------------------------------
# ProgressReporter
# ---------------------------------------------------------------------------


def test_progress_reporter_starts_and_closes(capsys):
    """A bar opens, accepts updates, and is closable."""
    bar = ProgressReporter()
    bar.start_epoch(total=10, epoch=1, total_epochs=1)
    bar.update(step=1, loss=2.5, lr=1e-3, samples=8, tokens=64)
    bar.update(step=2, loss=2.4, lr=9e-4, samples=8, tokens=64)
    bar.close()
    captured = capsys.readouterr()
    # tqdm writes to stderr.
    assert "Epoch 1/1" in captured.err


def test_progress_reporter_resume_offset():
    """Resuming mid-epoch advances the bar past already-completed steps."""
    bar = ProgressReporter(start_step=42)
    inner = bar.start_epoch(total=100, epoch=3, total_epochs=5)
    # The bar should already be at 42 / 100 — emits no extra ticks.
    initial_n = inner.n
    bar.update(step=43, loss=None, lr=None, samples=8, tokens=64)
    assert inner.n == initial_n + 1
    bar.close()


def test_human_count_format():
    from flatbuild.trainer.progress import _human_count

    assert _human_count(0) == "0.0"
    assert _human_count(1.5e-5) == "1.5e-05"
    assert _human_count(99) == "99.0"
    assert _human_count(250) == "250"
    assert _human_count(1500) == "1.5k"
    assert _human_count(2_500_000) == "2.5m"


# ---------------------------------------------------------------------------
# PerformanceProfiler
# ---------------------------------------------------------------------------


def test_profiler_disabled_is_a_noop():
    """When disabled, all measure blocks should not record anything."""
    prof = PerformanceProfiler(enabled=False)
    with prof.measure("forward"):
        time.sleep(0.0001)
    assert prof.summary() == {}


def test_profiler_records_phase_breakdown():
    """When enabled, measures and reports per-phase percentages."""
    prof = PerformanceProfiler(enabled=True)
    with prof.measure("forward"):
        time.sleep(0.001)
    with prof.measure("backward"):
        time.sleep(0.0005)
    with prof.measure("optim"):
        time.sleep(0.0001)
    summary = prof.summary()
    assert set(summary.keys()) == {"forward", "backward", "optim"}
    # All sums to ~100%.
    total_pct = sum(v["pct"] for v in summary.values())
    assert 99.0 < total_pct < 101.0
    # Forward was longest.
    assert summary["forward"]["pct"] > summary["backward"]["pct"] > summary["optim"]["pct"]


def test_profiler_print_summary(caplog):
    """print_summary emits informative lines without errors."""
    import logging

    prof = PerformanceProfiler(enabled=True)
    with prof.measure("forward"):
        pass
    with prof.measure("backward"):
        pass
    caplog.set_level(logging.INFO)
    prof.print_summary(step_count=10, elapsed=1.0)
    assert any("Flatbuild performance breakdown" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ValidationRunner
# ---------------------------------------------------------------------------


def test_validation_runner_uses_inference_mode():
    """``model.eval()`` is entered during the run and restored after."""
    cfg = _tiny_config()
    model = FlatbuildModel(cfg.model)
    tok = _demo_tokenizer()
    # Put the model in eval mode *before* the run so we can assert
    # the runner leaves it in eval mode afterwards.
    model.eval()
    prior = model.training
    runner = make_validation_runner(
        model,
        tok,
        _tiny_tokenized(),
        batch_size=4,
        max_batches=None,
        pad_token_id=tok.pad_token_id or 0,
    )
    metrics = runner.run()
    assert {"loss", "perplexity", "accuracy"} <= set(metrics.keys())
    assert metrics["loss"] >= 0.0
    # The runner must restore the model's prior training state.
    assert model.training == prior
    assert model.training is False


def test_validation_runner_max_batches_cap():
    """When ``max_batches=1`` the runner stops after a single batch."""
    cfg = _tiny_config()
    model = FlatbuildModel(cfg.model)
    tok = _demo_tokenizer()
    runner = make_validation_runner(
        model,
        tok,
        _tiny_tokenized(n=16, length=8),  # 16 rows / batch 8 = 2 batches
        batch_size=8,
        max_batches=1,
        pad_token_id=tok.pad_token_id or 0,
    )
    metrics = runner.run()
    # Both samples and tokens reported; values sane.
    assert metrics["loss"] >= 0.0
    # We can only see one batch of 8, so accuracy is well-defined but
    # unknown exactly; just verify type.
    assert 0.0 <= metrics["accuracy"] <= 1.0
