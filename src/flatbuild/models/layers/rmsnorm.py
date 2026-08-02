"""Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm with a learnable scale and configurable epsilon.

    Computes ``x / RMS(x) * weight`` where ``RMS(x) = sqrt(mean(x^2))``.
    """

    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        """Initialize RMSNorm.

        Args:
            dim: Channel dimension.
            eps: Numerical-stability epsilon.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to ``x``.

        Args:
            x: Tensor with last dim equal to ``dim``.

        Returns:
            Normalized tensor of the same shape.
        """
        in_dtype = x.dtype
        # Compute in float32 for numerical stability, cast back at the end.
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = x_f * rms * self.weight.float()
        return out.to(in_dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"
