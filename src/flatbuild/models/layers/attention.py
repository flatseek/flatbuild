"""Multi-head / Grouped-Query attention with RoPE and KV cache support."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from flatbuild.models.layers.rope import RotaryEmbedding, apply_rope


@dataclass
class AttentionConfig:
    """Static configuration for :class:`Attention`."""

    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    dropout: float = 0.0
    rope: RotaryEmbedding | None = None
    bias: bool = False
    sliding_window: int | None = None


class Attention(nn.Module):
    """Grouped-Query Attention.

    Q has ``n_heads`` heads; K and V have ``n_kv_heads`` heads (each KV
    head is repeated ``n_heads // n_kv_heads`` times). RoPE is applied
    to the rotated halves of the last axis. When ``past_kv`` is passed
    in, the present K/V is concatenated to it (for autoregressive
    inference with a KV cache).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int | None = None,
        rope_theta: float = 10000.0,
        max_seq_len: int = 4096,
        rope_scaling: dict | None = None,
        bias: bool = False,
        dropout: float = 0.0,
        sliding_window: int | None = None,
    ) -> None:
        """Initialize attention layers.

        Args:
            hidden_dim: Model hidden dimension.
            n_heads: Number of query heads.
            n_kv_heads: Number of key/value heads (GQA).
            head_dim: Per-head dimension. Defaults to ``hidden_dim // n_heads``.
            rope_theta: RoPE base.
            max_seq_len: Maximum expected sequence length.
            rope_scaling: Optional RoPE scaling dict.
            bias: Whether to include bias on projections.
            dropout: Attention dropout.
            sliding_window: Optional sliding-window length.
        """
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim or hidden_dim // n_heads
        if self.head_dim * n_heads != hidden_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal n_heads * head_dim ({n_heads * self.head_dim})"
            )

        self.q_proj = nn.Linear(hidden_dim, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(0.0)

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            base=rope_theta,
            max_seq_len=max_seq_len,
            scaling=rope_scaling,
        )
        self.sliding_window = sliding_window

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run attention.

        Args:
            x: Tensor of shape ``(B, T, hidden_dim)``.
            past_kv: Optional cached ``(K, V)`` from a previous step.
            attention_mask: Optional additive attention mask of
                shape ``(B, T_total)`` with values ``0`` / ``-inf``.

        Returns:
            ``(output, present_kv)`` where ``output`` has the same shape
            as ``x`` and ``present_kv`` is the new cache ``(K, V)``.
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE: apply only to the *current* T positions, offset by the
        # cache length so positions line up with the full sequence.
        past_len = past_kv[0].shape[2] if past_kv is not None else 0
        full_len = past_len + T
        cos_all, sin_all = self.rope(x, full_len)
        cos = cos_all[past_len:full_len]
        sin = sin_all[past_len:full_len]
        q, k = apply_rope(q, k, cos, sin)

        # KV cache concatenation: cache stores *post-RoPE* k so the
        # past rows retain the right position offsets.
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)

        # Repeat KV heads for GQA so we get ``(B, n_heads, T, head_dim)``.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention with optional causal mask.
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        T_q = q.shape[2]
        T_k = k.shape[2]
        # Build the causal mask. Query position ``i`` (in [0, T_q))
        # attends to key position ``j`` (in [0, T_k)) iff
        # ``j <= offset + i`` where ``offset = T_k - T_q``.
        offset = T_k - T_q
        q_idx = torch.arange(T_q, device=x.device).unsqueeze(1)
        k_idx = torch.arange(T_k, device=x.device).unsqueeze(0)
        keep = k_idx <= q_idx + offset   # bool, True => attend
        attn_scores = attn_scores.masked_fill(~keep, float("-inf"))

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        return out, present_kv
