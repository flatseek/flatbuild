"""Read a GGUF file and reconstruct a PyTorch-compatible state dict.

This module inverts the ``_hf_to_gguf_name`` mapping used by
:func:`flatbuild.exporters.gguf.GGUFExporter` so that a GGUF written by
flatbuild can be loaded, then re-exported (e.g. with a different
quantisation level).  It also extracts the model hyper-parameters from the
GGUF metadata so the caller can pass them to the exporter directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flatbuild.utils import get_logger

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)

# --------------------------------------------------------------------
# Reverse name mapping: GGUF canonical -> HuggingFace / Llama names
# --------------------------------------------------------------------


def _gguf_to_hf_name(name: str) -> str:
    """Invert ``_hf_to_gguf_name`` to convert GGUF names back to HF names.

    Unknown names are passed through unchanged.
    """
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name == "output.weight":
        return "lm_head.weight"
    prefix = "blk."
    if name.startswith(prefix):
        rest = name[len(prefix) :]
        idx, _, tail = rest.partition(".")
        head = f"model.layers.{idx}."
        if tail == "attn_q.weight":
            return head + "self_attn.q_proj.weight"
        if tail == "attn_k.weight":
            return head + "self_attn.k_proj.weight"
        if tail == "attn_v.weight":
            return head + "self_attn.v_proj.weight"
        if tail == "attn_output.weight":
            return head + "self_attn.o_proj.weight"
        if tail == "attn_norm.weight":
            return head + "input_layernorm.weight"
        if tail == "ffn_norm.weight":
            return head + "post_attention_layernorm.weight"
        if tail == "ffn_gate.weight":
            return head + "mlp.gate_proj.weight"
        if tail == "ffn_up.weight":
            return head + "mlp.up_proj.weight"
        if tail == "ffn_down.weight":
            return head + "mlp.down_proj.weight"
    return name


# --------------------------------------------------------------------
# GGUF -> HF state dict + model config
# --------------------------------------------------------------------

def load_gguf_state_dict(
    gguf_path: str | Path,
) -> tuple[dict[str, "torch.Tensor"], dict]:
    """Load a GGUF file and return a HF-compatible state dict + model config.

    Args:
        gguf_path: Path to a GGUF file (e.g. ``model.gguf``).

    Returns:
        ``(state_dict, model_config)`` where ``state_dict`` maps HF tensor
        names to ``torch.Tensor`` and ``model_config`` is a dict with
        keys compatible with :class:`flatbuild.config.ModelConfig`
        (``n_heads``, ``n_layers``, ``hidden_dim``, ``n_kv_heads``,
        ``context_length``, ``rope_theta``).
    """
    import numpy as np
    import torch

    from gguf import GGUFReader

    gguf_path = Path(gguf_path)
    logger.info(f"Reading GGUF from {gguf_path}")

    reader = GGUFReader(str(gguf_path))

    # Extract hyper-parameters from GGUF metadata.
    def _i(key: str) -> int:
        try:
            return int(reader.fields[key].parts[-1].tolist()[0])
        except Exception:
            return 0

    def _f(key: str) -> float:
        try:
            return float(reader.fields[key].parts[-1].tolist()[0])
        except Exception:
            return 0.0

    n_layers = _i("llama.block_count")
    n_heads = _i("llama.attention.head_count")
    n_kv = _i("llama.attention.head_count_kv")
    hidden_dim = _i("llama.embedding_length")
    ffn_dim = _i("llama.feed_forward_length")
    ctx_len = _i("llama.context_length")
    rope_theta = _f("llama.rope.freq_base")

    model_config = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "hidden_dim": hidden_dim,
        "n_kv_heads": n_kv or max(1, n_heads // 2),
        "ffn_dim": ffn_dim,
        "context_length": ctx_len,
        "rope_theta": rope_theta,
    }
    logger.info(
        f"GGUF config: {n_layers}L, hidden={hidden_dim}, heads={n_heads}/{n_kv}"
    )

    from gguf import GGMLQuantizationType

    # Build state dict.
    state_dict: dict[str, torch.Tensor] = {}
    skipped = 0
    for tensor in reader.tensors:
        hf_name = _gguf_to_hf_name(tensor.name)
        try:
            qtype = tensor.tensor_type
            if qtype not in (GGMLQuantizationType.F16, GGMLQuantizationType.F32):
                logger.warning(
                    f"Skipping tensor {tensor.name}: quantized types "
                    f"({qtype.name}) require decompression — "
                    f"use llama.cpp to dequantize first."
                )
                skipped += 1
                continue

            dtype = np.float32 if qtype == GGMLQuantizationType.F32 else np.float16
            # Read raw bytes at the stored offset into the correct dtype.
            raw_bytes = gguf_path.read_bytes()[tensor.data_offset : tensor.data_offset + tensor.n_bytes]
            arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(tensor.shape)
            # GGUF stores embedding/lm_head as (hidden, vocab), PyTorch/HF uses (vocab, hidden)
            if hf_name in ("model.embed_tokens.weight", "lm_head.weight"):
                arr = arr.T.copy()
            state_dict[hf_name] = torch.from_numpy(arr.copy())
        except Exception as exc:
            logger.warning(f"Skipping tensor {tensor.name}: {exc}")
            skipped += 1

    if skipped:
        logger.warning(
            f"Skipped {skipped}/{len(reader.tensors)} tensors "
            f"(quantised). Load with llama.cpp to dequantise, or re-export "
            f"from source weights."
        )
    logger.info(f"Loaded {len(state_dict)} tensors from GGUF.")

    return state_dict, model_config
