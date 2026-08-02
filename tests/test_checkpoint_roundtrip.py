"""Checkpoint save→load→generate round-trip regression test.

Guards the exact bug this file was written for: ``FlatbuildModel.
load_state_dict_llama`` used to hand Llama-format keys straight to
``load_state_dict`` without remapping them to the model's internal
parameter names. Because the CLI loads with ``strict=False``, the
mismatch was silently swallowed and every attention/MLP weight stayed
at its random initialisation — so generation from a *loaded* checkpoint
produced garbage while the in-memory model generated fine.

The test proves that after

    train → save checkpoint → load checkpoint → generate

the loaded model is bit-identical to the trained model and produces
the exact same tokens.
"""

from __future__ import annotations

import time

import pytest
import torch

from flatbuild.checkpoint.manager import CheckpointManager
from flatbuild.config import (
    ChatTemplateConfig,
    CheckpointConfig,
    DatasetConfig,
    EarlyStoppingConfig,
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
from flatbuild.datasets.base import ConversationSample
from flatbuild.models import FlatbuildModel
from flatbuild.tokenizers.bpe import BPETokenizer
from flatbuild.tokenizers.template import build_chat_template
from flatbuild.trainer.trainer import FlatbuildTrainer


@pytest.mark.slow
def test_checkpoint_roundtrip_generation_is_identical(tmp_path):
    """Train, checkpoint, reload — the loaded model must match the trained one."""
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

    cfg = FlatBuildConfig(
        name="roundtrip-2plus2",
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

    run_dir = tmp_path / "run"
    model = FlatbuildModel(cfg.model)
    trainer = FlatbuildTrainer(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        samples=[sample] * 4,
        val_samples=[],
        run_dir=run_dir,
    )

    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"round-trip train took {elapsed:.1f}s — config too big"

    prompt_text = chat_template.render(
        [("user", "2+2?")], add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text)
    assert prompt_ids, "prompt tokenized to empty"
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # ---- In-memory reference generation (must not regress) ----------
    model.eval()
    with torch.inference_mode():
        ref_out = model.generate(
            input_ids,
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    ref_new = ref_out[0, input_ids.shape[1] :].tolist()
    ref_text = tokenizer.decode([int(i) for i in ref_new]).replace(tokenizer.eos_token, "").strip()
    assert "4" in ref_text, f"in-memory reference did not generate '4': {ref_text!r}"

    # ---- Save the final checkpoint, then reload it ------------------
    final_dir = run_dir / "checkpoints" / "final"
    assert final_dir.exists(), f"final checkpoint not saved at {final_dir}"

    bundle = CheckpointManager.load(final_dir)
    cfg_loaded = bundle["config"]
    assert cfg_loaded is not None
    model2 = FlatbuildModel(cfg_loaded.model)
    load_result = model2.load_state_dict_llama(bundle["model_state_dict"], strict=False)

    # The core regression: every weight must load — nothing silently skipped.
    assert not load_result.missing_keys, (
        f"checkpoint load dropped weights: {load_result.missing_keys}"
    )
    assert not load_result.unexpected_keys, (
        f"checkpoint load ignored weights: {load_result.unexpected_keys}"
    )

    # ---- Loaded model is bit-identical to the trained model ---------
    sd_train = model.state_dict_llama()
    sd_loaded = model2.state_dict_llama()
    assert set(sd_train) == set(sd_loaded)
    for key in sd_train:
        assert torch.equal(sd_train[key], sd_loaded[key]), f"weight drift on {key}"

    # Tied embeddings must survive the round-trip.
    assert torch.equal(model2.embed_tokens.weight.data, model2.lm_head.weight.data)

    # ---- Generation from the loaded model == in-memory generation ----
    model2.eval()
    with torch.inference_mode():
        loaded_out = model2.generate(
            input_ids,
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    assert torch.equal(ref_out, loaded_out), (
        "loaded model generated different tokens than the in-memory model"
    )
    loaded_text = tokenizer.decode(
        [int(i) for i in loaded_out[0, input_ids.shape[1] :].tolist()]
    ).replace(tokenizer.eos_token, "").strip()
    assert "4" in loaded_text, f"loaded checkpoint did not generate '4': {loaded_text!r}"

    # ---- KV cache must not change the answer (reference no-cache path) ----
    with torch.inference_mode():
        no_cache_out = model2.generate(
            input_ids,
            max_new_tokens=6,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    assert torch.equal(loaded_out, no_cache_out), (
        "generate(use_cache=True) diverges from generate(use_cache=False)"
    )
