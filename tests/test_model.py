"""Tests for the Transformer model, attention, RoPE, RMSNorm, SwiGLU."""

from __future__ import annotations

import pytest
import torch

from flatbuild.config import ModelConfig
from flatbuild.models import FlatbuildModel
from flatbuild.models.layers import (
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    apply_rope,
    rotate_half,
)
from flatbuild.models.layers.attention import Attention


def test_rmsnorm_basic():
    """RMSNorm keeps the output shape and rescales input."""
    norm = RMSNorm(8)
    x = torch.randn(2, 3, 8)
    out = norm(x)
    assert out.shape == x.shape
    # Output RMS per row should be ~1.
    rms = out.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_swiglu_shape():
    """SwiGLU block returns the same hidden size."""
    block = SwiGLU(hidden_dim=16, ffn_dim=64)
    x = torch.randn(2, 3, 16)
    out = block(x)
    assert out.shape == x.shape


def test_rotate_half_round_trip():
    """rotate_half followed by a rotated inverse recovers the input (up to sign)."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rotated = rotate_half(x)
    # Rotation is involutive: rotate(rotate(x)) == -x.
    again = rotate_half(rotated)
    assert torch.allclose(again, -x)


def test_apply_rope_shape_preserved():
    """apply_rope keeps the input shape."""
    head_dim = 8
    seq_len = 4
    rope = RotaryEmbedding(head_dim=head_dim, base=10000.0, max_seq_len=seq_len)
    cos, sin = rope(torch.zeros(1, seq_len, head_dim), seq_len)
    q = torch.randn(2, 4, seq_len, head_dim)
    k = torch.randn(2, 4, seq_len, head_dim)
    qr, kr = apply_rope(q, k, cos, sin)
    assert qr.shape == q.shape
    assert kr.shape == k.shape


def test_attention_with_cache():
    """Attention with a KV cache should match the prefill output for cached prefix."""
    attn = Attention(hidden_dim=16, n_heads=4, n_kv_heads=2, head_dim=4, max_seq_len=32)
    attn.eval()
    x_full = torch.randn(1, 8, 16)
    with torch.no_grad():
        out_full, _ = attn(x_full)

    # Prefill first 7 tokens, then decode the 8th. The 8th-token
    # output must equal the same-token output from the full-8 prefill.
    past_in = x_full[:, :7, :]
    _, kvcache = attn(past_in)
    new_token = x_full[:, 7:8, :]
    out_step2, _ = attn(new_token, past_kv=kvcache)
    assert torch.allclose(out_full[:, 7:8, :], out_step2, atol=1e-5)


def test_model_forward_and_state_dict_keys():
    """Forward returns (B, T, V) logits and Llama-style state dict has standard keys."""
    cfg = ModelConfig(vocab_size=64, n_layers=2, n_heads=2, n_kv_heads=1, hidden_dim=32, ffn_dim=128, context_length=16)
    model = FlatbuildModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(x, labels=x)
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.loss is not None
    sd = model.state_dict_llama()
    # Llama/HF naming.
    assert "model.embed_tokens.weight" in sd
    assert "model.layers.0.self_attn.q_proj.weight" in sd
    assert "model.layers.0.mlp.gate_proj.weight" in sd
    assert "model.norm.weight" in sd
    assert "lm_head.weight" in sd
    # Round-trip load with strict=False should not throw.
    cfg2 = ModelConfig(vocab_size=64, n_layers=2, n_heads=2, n_kv_heads=1, hidden_dim=32, ffn_dim=128, context_length=16)
    model2 = FlatbuildModel(cfg2)
    model2.load_state_dict_llama(sd, strict=False)
