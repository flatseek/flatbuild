"""Rotary Position Embeddings (RoPE)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class RopeConfig:
    """Configuration for :class:`RotaryEmbedding`."""

    head_dim: int
    base: float = 10000.0
    scaling: dict | None = None

    def __post_init__(self) -> None:
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {self.head_dim}")


class RotaryEmbedding(nn.Module):
    """Sinusoidal rotary embeddings (Su et al., 2021).

    The cache precomputes ``cos``/``sin`` tables for every (position,
    dim) pair up to ``max_seq_len``. During attention we apply them as
    a per-pair rotation.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        max_seq_len: int = 4096,
        scaling: dict | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the rotary cache.

        Args:
            head_dim: Per-head dimension (``hidden_dim // n_heads``).
            base: Exponential base of the sinusoid.
            max_seq_len: Maximum expected sequence length.
            scaling: Optional scaling dict (e.g. ``{"type": "linear",
                "factor": 2.0}``). Currently ``None`` — Linear scaling
                could be implemented here.
            device: Device on which to place the cache.
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.scaling = scaling or {}

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        # Linear scaling (LLaMA-style): simply rescale positions.
        if scaling and scaling.get("type") == "linear":
            factor = float(scaling.get("factor", 1.0))
            inv_freq = inv_freq / factor

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len, device)

    def _build_cache(self, max_seq_len: int, device: torch.device | str | None) -> None:
        """Pre-compute cos / sin for positions up to ``max_seq_len``."""
        t = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Different from the original implementation: we rotate pairs
        # at positions ``[i, i + head_dim/2]`` rather than interleaving.
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` sliced to the needed ``seq_len``.

        Args:
            x: Query or key tensor (used only to infer dtype/device).
            seq_len: Number of tokens in the current sequence.

        Returns:
            Tuple ``(cos, sin)`` of shape ``(seq_len, head_dim)``.
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len, x.device)
        cos = self.cos_cached[:seq_len].to(x.dtype)
        sin = self.sin_cached[:seq_len].to(x.dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of ``x`` along the last axis.

    Equivalent to RoPE's per-pair rotation.
    """
    head = x.shape[-1] // 2
    x1, x2 = x[..., :head], x[..., head:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to ``q`` and ``k``.

    Args:
        q: Query tensor of shape ``(... , seq, head_dim)``.
        k: Key tensor of shape ``(... , seq, head_dim)``.
        cos, sin: Tensors of shape ``(seq, head_dim)``.

    Returns:
        Tuple of rotated ``(q, k)``.
    """
    # Add batch dim so broadcasting works for non-batched calls.
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
