"""Round-trip tests for the new GGUF and FWG exporters."""

from __future__ import annotations

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
    SchedulerConfig,
    TokenizerConfig,
    TrainerConfig,
)
from flatbuild.models import FlatbuildModel
from flatbuild.tokenizers.bpe import BPETokenizer


def _tiny_config() -> FlatBuildConfig:
    return FlatBuildConfig(
        name="gguf-fwg-test",
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
        optimizer=OptimizerConfig(),
        scheduler=SchedulerConfig(),
        trainer=TrainerConfig(),
        checkpoint=CheckpointConfig(),
        export=ExportConfig(),
        generate=GenerateConfig(),
    )


def test_gguf_round_trip(tmp_path):
    """GGUF written by Flatbuild re-loads via gguf.GGUFReader with matching tensors.

    Tensor names use the GGUF convention (``token_embd.weight`` etc.)
    so llama.cpp / flatrun can find them.
    """
    pytest.importorskip("gguf")
    from gguf import GGUFReader

    from flatbuild.exporters.gguf import GGUFExporter, _hf_to_gguf_name

    cfg = _tiny_config()
    model = FlatbuildModel(cfg.model)
    out = GGUFExporter(copy_tokenizer=False).export(model, tmp_path / "out")
    gguf_path = out / "model.gguf"
    assert gguf_path.exists()

    reader = GGUFReader(str(gguf_path))
    # Names on disk are GGUF-style; map expected HF names to verify
    # the writer applied the right transform.
    expected_names = {_hf_to_gguf_name(n) for n in model.state_dict_llama().keys()}
    actual = {t.name for t in reader.tensors}
    assert expected_names == actual, (
        f"GGUF tensor names mismatch.\n"
        f"  expected: {sorted(expected_names)}\n"
        f"  actual:   {sorted(actual)}"
    )
    # Sanity check: one tensor should be the GGUF-style embed name.
    assert "token_embd.weight" in actual
    # Tensor shapes/values match.
    pick_hf = "model.embed_tokens.weight"
    pick_gguf = "token_embd.weight"
    t = next(t for t in reader.tensors if t.name == pick_gguf)
    arr = t.data
    src = model.state_dict_llama()[pick_hf].cpu().numpy().astype("float32", copy=False)
    assert arr.shape == src.shape
    assert (arr == src).all()


def test_gguf_tensor_names_are_canonical(tmp_path):
    """GGUF tensor names must match the llama.cpp / flatrun canonical set.

    Regression: ``self_attn.o_proj.weight`` was written as
    ``blk.N.attn_o.weight``, but llama.cpp and flatrun both expect the
    output projection under ``blk.N.attn_output.weight`` — flatrun's
    GGUF reader leaves ``attn_o`` untranslated and the o_proj lookup
    fails at the first decoder block.
    """
    pytest.importorskip("gguf")
    from gguf import GGUFReader

    from flatbuild.exporters.gguf import GGUFExporter

    cfg = _tiny_config()
    model = FlatbuildModel(cfg.model)
    out = GGUFExporter(copy_tokenizer=False).export(model, tmp_path / "out")
    reader = GGUFReader(str(out / "model.gguf"))
    actual = {t.name for t in reader.tensors}

    # Canonical llama.cpp names (see convert_hf_to_gguf.py / llama-arch).
    canonical = {
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        *{f"blk.{i}.attn_norm.weight" for i in range(cfg.model.n_layers)},
        *{f"blk.{i}.attn_q.weight" for i in range(cfg.model.n_layers)},
        *{f"blk.{i}.attn_k.weight" for i in range(cfg.model.n_layers)},
        *{f"blk.{i}.attn_v.weight" for i in range(cfg.model.n_layers)},
        *{f"blk.{i}.attn_output.weight" for i in range(cfg.model.n_layers)},
        *{f"blk.{i}.ffn_norm.weight" for i in range(cfg.model.n_layers)},
        *{f"blk.{i}.ffn_gate.weight" for i in range(cfg.model.n_layers)},
        *{f"blk.{i}.ffn_up.weight" for i in range(cfg.model.n_layers)},
        *{f"blk.{i}.ffn_down.weight" for i in range(cfg.model.n_layers)},
    }
    assert actual == canonical, (
        f"GGUF tensor names are not canonical.\n"
        f"  missing: {sorted(canonical - actual)}\n"
        f"  extra:   {sorted(actual - canonical)}"
    )
    # The non-canonical alias must never appear.
    assert not any(n.endswith("attn_o.weight") for n in actual)


def test_gguf_tokenizer_attached_flat(tmp_path):
    """GGUF exporter attaches tokenizer at root when one is passed."""
    pytest.importorskip("gguf")
    from flatbuild.exporters.gguf import GGUFExporter

    cfg = _tiny_config()
    model = FlatbuildModel(cfg.model)
    tok = BPETokenizer.train(["a b c"], vocab_size=32, min_frequency=1)
    tok_dir = tmp_path / "tok"
    tok.save(tok_dir)

    out = GGUFExporter(copy_tokenizer=True).export(
        model, tmp_path / "out", tokenizer_dir=tok_dir
    )
    assert (out / "model.gguf").exists()
    assert (out / "tokenizer.json").exists()
    assert (out / "tokenizer_config.json").exists()
    assert not (out / "tokenizer").exists()


def test_gguf_embeds_tokenizer_inside_file(tmp_path):
    """The BPE tokenizer is embedded in the GGUF file as metadata, so flatrun/llama.cpp
    can use the .gguf file with no sidecar."""
    pytest.importorskip("gguf")
    from gguf import GGUFReader
    from flatbuild.exporters.gguf import GGUFExporter

    cfg = _tiny_config()
    model = FlatbuildModel(cfg.model)
    tok = BPETokenizer.train(["a b c d e f g h i j"], vocab_size=64, min_frequency=1)
    tok_dir = tmp_path / "tok"
    tok.save(tok_dir)

    out = GGUFExporter(copy_tokenizer=True).export(
        model, tmp_path / "out", tokenizer_dir=tok_dir
    )
    reader = GGUFReader(str(out / "model.gguf"))
    # The BPE tokenizer is encoded as a set of GGUF kv pairs.
    kv_keys = {k for k in reader.fields.keys()}
    assert "tokenizer.ggml.model" in kv_keys
    assert "tokenizer.ggml.tokens" in kv_keys
    assert "tokenizer.ggml.bos_token_id" in kv_keys
    assert "tokenizer.ggml.eos_token_id" in kv_keys
    assert "tokenizer.ggml.unknown_token_id" in kv_keys
    # And the vocab has the expected number of tokens.
    n_tokens = len(reader.fields["tokenizer.ggml.tokens"].contents())
    assert n_tokens == tok.vocab_size, (
        f"GGUF-embedded vocab has {n_tokens} tokens, "
        f"but the BPE tokenizer was built with {tok.vocab_size}"
    )


def test_fwg_round_trip(tmp_path):
    """FWG written by Flatbuild re-loads back via flatweight.WeightFS with matching tensors."""
    pytest.importorskip("flatweight")

    from flatbuild.exporters.fwg import FWGExporter

    cfg = _tiny_config()
    model = FlatbuildModel(cfg.model)
    out = FWGExporter(copy_tokenizer=False, tile_size=64).export(model, tmp_path / "out")
    fwg_path = out / "model.fwg"
    assert fwg_path.exists()
    assert fwg_path.stat().st_size > 0

    from flatbuild.exporters._fwg_reader import load_fwg_state_dict

    sd, meta = load_fwg_state_dict(fwg_path)
    expected = model.state_dict_llama()
    assert set(sd.keys()) == set(expected.keys())
    assert meta["tensor_count"] == len(expected)
    pick = "model.embed_tokens.weight"
    assert torch.allclose(sd[pick].to(torch.float32), expected[pick].cpu().to(torch.float32), atol=1e-6)


def test_fwg_tokenizer_attached_flat(tmp_path):
    """FWG exporter attaches tokenizer at root when one is passed."""
    pytest.importorskip("flatweight")

    from flatbuild.exporters.fwg import FWGExporter

    cfg = _tiny_config()
    model = FlatbuildModel(cfg.model)
    tok = BPETokenizer.train(["x y z"], vocab_size=32, min_frequency=1)
    tok_dir = tmp_path / "tok"
    tok.save(tok_dir)

    out = FWGExporter(copy_tokenizer=True, tile_size=64).export(
        model, tmp_path / "out_fwg", tokenizer_dir=tok_dir
    )
    assert (out / "model.fwg").exists()
    assert (out / "tokenizer.json").exists()
    assert (out / "tokenizer_config.json").exists()
    assert not (out / "tokenizer").exists()
