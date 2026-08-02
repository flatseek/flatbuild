"""Feed-forward activations.

- :class:`SwiGLU` — gated SiLU/GLU (Shazeer, 2020). Used in LLaMA,
  Mistral, Qwen2 etc.
- :class:`ReLU2` — alternative used by the PaLM paper.
- :class:`GELU` — standard GPT-style feed-forward.
"""

from __future__ import annotations

import torch
from torch import nn


class SwiGLU(nn.Module):
    """SwiGLU MLP with gated SiLU: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, hidden_dim: int, ffn_dim: int, bias: bool = False) -> None:
        """Initialize the gated MLP.

        Args:
            hidden_dim: Input/output channel size.
            ffn_dim: Inner channel size.
            bias: Whether to use bias terms on linear projections.
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the SwiGLU block.

        Args:
            x: Tensor of shape ``(..., hidden_dim)``.

        Returns:
            Tensor of the same shape with the same dtype as ``x``.
        """
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class ReLU2(nn.Module):
    """``relu(x)^2`` activation followed by a down-projection."""

    def __init__(self, hidden_dim: int, ffn_dim: int, bias: bool = False) -> None:
        """Initialize the ReLU^2 MLP."""
        super().__init__()
        self.up = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.down = nn.Linear(ffn_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ReLU^2 block."""
        h = torch.nn.functional.relu(self.up(x))
        return self.down(h * h)


class GELU(nn.Module):
    """Standard GPT-style GELU FFN: ``down(gelu(up(x)))``."""

    def __init__(self, hidden_dim: int, ffn_dim: int, bias: bool = False) -> None:
        """Initialize the GELU FFN."""
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the GELU FFN."""
        return self.down_proj(torch.nn.functional.gelu(self.up_proj(x)))


def build_activation(
    name: str,
    hidden_dim: int,
    ffn_dim: int,
) -> nn.Module:
    """Construct an activation MLP by name.

    Args:
        name: One of ``"swiglu"``, ``"relu2"``, ``"gelu"``.
        hidden_dim: Hidden size.
        ffn_dim: Inner FFN size.

    Returns:
        An ``nn.Module`` instance.
    """
    if name == "swiglu":
        return SwiGLU(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
    if name == "relu2":
        return ReLU2(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
    if name == "gelu":
        return GELU(hidden_dim=hidden_dim, ffn_dim=ffn_dim)
    raise ValueError(f"Unknown activation: {name}")
