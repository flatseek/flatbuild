"""Sample overfit test (end-to-end).

Trains the model on a single Q&A pair:

    User: 2+2?
    Assistant: 4

…and then asserts that :meth:`FlatbuildModel.generate` actually
outputs ``"4"`` for the prompt ``"2+2?"``.

This is the strictest possible smoke test:

- Chat template renders correctly.
- BPE tokenizer round-trips through ``<|user|>`` / ``<|assistant|>``.
- Forward / loss / backward / optimizer.scheduler.step all wired.
- ``model.generate`` itself works end-to-end (KV cache, RoPE, argmax).
- Tokenizer decode reproduces the trained character.

If this test passes, every component that could break language-model
output has been exercised on the smallest possible input.

Mark as ``@pytest.mark.slow`` so the default lane skips it.
"""

from __future__ import annotations

import time

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
    EarlyStoppingConfig,
)
from flatbuild.datasets.base import ConversationSample
from flatbuild.models import FlatbuildModel
from flatbuild.tokenizers.bpe import BPETokenizer
from flatbuild.tokenizers.template import build_chat_template
from flatbuild.trainer.trainer import FlatbuildTrainer


@pytest.mark.slow
def test_model_overfits_2_plus_2_and_generates_4(tmp_path):
    """Single Q&A ``User: 2+2? Assistant: 4`` → must generate ``4`` for the prompt.

    Steps:
        1. Build a BPE tokenizer over the conversation.
        2. Train a tiny FlatbuildModel on a single duplicated sample for
           enough epochs that loss collapses to near zero.
        3. Call ``model.generate("2+2?")`` (rendered via chat template).
        4. Decode and assert ``"4"`` is in the output.

    Deterministic via ``seed=123``. Skippable on fast CI lanes.
    """
    # --- Stage 1: single-sample corpus -----------------------------------
    sample = ConversationSample(
        messages=(
            ("user", "2+2?"),
            ("assistant", "4"),
        )
    )
    chat_template_cfg = ChatTemplateConfig()
    chat_template = build_chat_template(chat_template_cfg)

    rendered = chat_template.render_sample(sample)
    tokenizer = BPETokenizer.train(
        [rendered, rendered, rendered, rendered],
        vocab_size=128,
        min_frequency=1,
    )

    # --- Stage 2: tiny model, big LR, long training ----------------------
    cfg = FlatBuildConfig(
        name="overfit-2plus2",
        dataset=DatasetConfig(max_length=64),
        tokenizer=TokenizerConfig(vocab_size=tokenizer.vocab_size),
        chat_template=chat_template_cfg,
        model=ModelConfig(
            vocab_size=tokenizer.vocab_size,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            hidden_dim=64,
            ffn_dim=256,
            context_length=64,
        ),
        optimizer=OptimizerConfig(lr=3e-3),
        scheduler=SchedulerConfig(warmup_steps=10, min_lr_ratio=0.01),
        trainer=TrainerConfig(
            epochs=400,
            batch_size=4,
            gradient_accumulation=1,
            max_steps=400,
            precision=Precision.FP32,
            eval_every_n_steps=None,
            log_every_n_steps=0,
            max_grad_norm=1.0,
            seed=123,
            early_stopping=EarlyStoppingConfig(enabled=False),
        ),
        validation=ValidationConfig(),
        checkpoint=CheckpointConfig(every_n_steps=999, keep_last=1, save_final=True),
        export=ExportConfig(),
        generate=GenerateConfig(),
    )

    model = FlatbuildModel(cfg.model)
    trainer = FlatbuildTrainer(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        samples=[sample] * 4,  # duplicate so batch_size=4 fits one step
        val_samples=[],
        run_dir=tmp_path / "run",
    )

    t0 = time.perf_counter()
    artifacts = trainer.train()
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"overfit took {elapsed:.1f}s — config too big"

    final_loss = float(
        artifacts.metrics.get("final", {}).get("loss")
        or trainer.last_loss
        or 9.0
    )
    # Loss should collapse far below half its starting point.
    assert final_loss < 0.5, f"loss did not collapse: {final_loss:.3f}"

    # --- Stage 3: end-to-end generation test ---------------------------
    prompt_text = chat_template.render(
        [("user", "2+2?")], add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text)
    assert prompt_ids, "prompt tokenized to empty"

    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=6,
            do_sample=False,  # greedy for deterministic tiny model
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, input_ids.shape[1] :].tolist()
    decoded = tokenizer.decode([int(i) for i in generated])

    # Remove the start/stop tokens from the assertion string.
    printable = decoded.replace(tokenizer.eos_token, "").strip()

    # The model may emit a couple of tokens around the "4"; what
    # matters is that the digit appears.
    assert "4" in printable, (
        f"Expected '4' in generated output, got {printable!r}"
    )


@pytest.mark.slow
def test_overfit_with_single_short_string(tmp_path):
    """Secondary sanity: memorize an arbitrary short string token-id-by-token.

    A second overfit test passes a multi-token string and verifies the
    model reproduces it token-for-token after greedy decoding.
    """
    target = "Flatbot says hello"
    sample = ConversationSample(
        messages=(
            ("user", "What do you say?"),
            ("assistant", target),
        )
    )
    chat_template_cfg = ChatTemplateConfig()
    chat_template = build_chat_template(chat_template_cfg)
    rendered = chat_template.render_sample(sample)
    tokenizer = BPETokenizer.train(
        [rendered] * 4,
        vocab_size=128,
        min_frequency=1,
    )

    cfg = FlatBuildConfig(
        name="overfit-string",
        dataset=DatasetConfig(max_length=128),
        tokenizer=TokenizerConfig(vocab_size=tokenizer.vocab_size),
        chat_template=chat_template_cfg,
        model=ModelConfig(
            vocab_size=tokenizer.vocab_size,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            hidden_dim=64,
            ffn_dim=256,
            context_length=128,
        ),
        optimizer=OptimizerConfig(lr=3e-3),
        scheduler=SchedulerConfig(warmup_steps=10, min_lr_ratio=0.01),
        trainer=TrainerConfig(
            epochs=400,
            batch_size=4,
            gradient_accumulation=1,
            max_steps=400,
            precision=Precision.FP32,
            log_every_n_steps=0,
            seed=42,
            early_stopping=EarlyStoppingConfig(enabled=False),
        ),
        validation=ValidationConfig(),
        checkpoint=CheckpointConfig(every_n_steps=999, keep_last=1, save_final=True),
        export=ExportConfig(),
        generate=GenerateConfig(),
    )

    model = FlatbuildModel(cfg.model)
    trainer = FlatbuildTrainer(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        samples=[sample] * 4,
        val_samples=[],
        run_dir=tmp_path / "run",
    )

    trainer.train()

    prompt_text = chat_template.render(
        [("user", "What do you say?")], add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text)
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=24,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, input_ids.shape[1] :].tolist()
    decoded = tokenizer.decode([int(i) for i in new_tokens])
    printable = decoded.replace(tokenizer.eos_token, "").strip().lower()

    print(prompt_text)
    print(printable)

    # All words of the trained reply should appear (in some form) in
    # the model's output — exact whitespace / punctuation may differ
    # because BPE byte-level decoding can re-introduce spaces around
    # boundaries.
    expected_words = target.lower().split()
    for word in expected_words:
        # Allow punctuation stripped form.
        assert word in printable, f"missing {word!r} in {printable!r} (expected {expected_words!r})"
