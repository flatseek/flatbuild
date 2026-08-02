"""Sub-modules of the decoder-only Transformer."""

from flatbuild.models.layers.attention import Attention, AttentionConfig
from flatbuild.models.layers.block import DecoderBlock
from flatbuild.models.layers.rmsnorm import RMSNorm
from flatbuild.models.layers.rope import RotaryEmbedding, apply_rope, rotate_half
from flatbuild.models.layers.swiglu import (
    GELU,
    ReLU2,
    SwiGLU,
    build_activation,
)

__all__ = [
    "Attention",
    "AttentionConfig",
    "DecoderBlock",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rope",
    "rotate_half",
    "SwiGLU",
    "ReLU2",
    "GELU",
    "build_activation",
]
