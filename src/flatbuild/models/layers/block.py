"""Single Transformer decoder block."""

from __future__ import annotations

import torch
from torch import nn

from flatbuild.models.layers.attention import Attention
from flatbuild.models.layers.rmsnorm import RMSNorm
from flatbuild.models.layers.swiglu import SwiGLU, ReLU2, GELU


class DecoderBlock(nn.Module):
    """One decoder block: pre-norm attention + pre-norm MLP."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int | None,
        ffn_dim: int,
        rope_theta: float,
        max_seq_len: int,
        rope_scaling: dict | None,
        norm: str = "rmsnorm",
        activation: str = "swiglu",
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        """Initialize a single decoder block.

        Args:
            hidden_dim: Hidden size.
            n_heads: Number of attention heads.
            n_kv_heads: Number of KV heads.
            head_dim: Per-head dimension.
            ffn_dim: FFN inner size.
            rope_theta: RoPE base.
            max_seq_len: Max expected seq length.
            rope_scaling: Optional RoPE scaling.
            norm: Normalization layer name.
            activation: Activation name (``swiglu`` / ``relu2`` / ``gelu``).
            dropout: Dropout probability.
            bias: Whether to use bias on projections.
        """
        super().__init__()
        self.attn = Attention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
            bias=bias,
            dropout=dropout,
        )
        self.mlp = _build_mlp(activation, hidden_dim, ffn_dim, bias=bias)
        self.norm_1 = _build_norm(norm, hidden_dim)
        self.norm_2 = _build_norm(norm, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run attention + MLP with pre-norm residuals.

        Args:
            x: Tensor of shape ``(B, T, hidden_dim)``.
            past_kv: Optional KV cache.
            attention_mask: Optional additive mask.

        Returns:
            ``(output, present_kv)``.
        """
        # Pre-norm attention (one forward, then residual).
        attn_out, present = self.attn(
            self.norm_1(x), past_kv=past_kv, attention_mask=attention_mask
        )
        h = x + attn_out
        # Pre-norm MLP (also with residual).
        h = h + self.mlp(self.norm_2(h))
        return h, present


def _build_norm(name: str, dim: int) -> nn.Module:
    if name == "rmsnorm":
        return RMSNorm(dim)
    if name == "layernorm":
        from torch.nn import LayerNorm

        return LayerNorm(dim)
    raise ValueError(f"Unknown norm: {name}")


def _build_mlp(name: str, hidden_dim: int, ffn_dim: int, *, bias: bool) -> nn.Module:
    if name == "swiglu":
        return SwiGLU(hidden_dim, ffn_dim, bias=bias)
    if name == "relu2":
        return ReLU2(hidden_dim, ffn_dim, bias=bias)
    if name == "gelu":
        return GELU(hidden_dim, ffn_dim, bias=bias)
    raise ValueError(f"Unknown activation: {name}")
