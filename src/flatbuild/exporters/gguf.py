"""Export a Flatbuild model to GGUF (llama.cpp's binary format).

This exporter uses the ``gguf`` Python package (pure-Python, official
ggml-org library). If you only need a standalone exporter install::

    pip install -e ".[gguf]"

Only F32 dtypes are written today. Quantization is intentionally out of
scope — Flatbuild trainers carry whatever precision the user picked, and
no post-training quantization is implemented yet (see build.md roadmap).

Tensor name mapping
-------------------

Flatbuild's internal state dict uses HuggingFace / Llama-style names
(``model.embed_tokens.weight``, ``blk.0.attn_q.weight``). GGUF consumers
(llama.cpp, flatrun) expect GGUF-style names (``token_embd.weight``,
``blk.0.attn_q.weight``). The mapping is applied on write so the
output file is consumable by every standard GGUF tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from flatbuild.config import ExportConfig
from flatbuild.exporters.base import Exporter
from flatbuild.models import FlatbuildModel
from flatbuild.tokenizers import build_chat_template
from flatbuild.tokenizers.template import to_flatrun_jinja
from flatbuild.utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time guard
    from gguf import GGMLQuantizationType

    import numpy as np

logger = get_logger(__name__)


def _resolve_quant_type(
    quant: str,
    ggml: "GGMLQuantizationType",
) -> tuple["GGMLQuantizationType", bool]:
    """Map a user-facing quant string to a GGMLQuantizationType.

    Args:
        quant: One of ``f16``, ``f32`` (identity) or a GGUF quant name
            (``q4_0``, ``q4_1``, ``q5_0``, ``q8_0``, ``q2_k``, ``q3_k``,
            ``q4_k``, ``q5_k``, ``q6_k``, ``q8_k``).
        ggml: The ``gguf.GGMLQuantizationType`` enum to look up.

    Returns:
        A ``(dtype, flag)`` tuple where ``flag`` is ``True`` when the dtype
        is a real quantization (not f16/f32).

    Raises:
        ValueError: If ``quant`` is not a known type.
    """
    name = quant.strip().upper()
    try:
        dtype = ggml[name]
    except KeyError:
        allowed = sorted(
            q.name
            for q in ggml
            if q.name
            in {"F16", "F32", "Q4_0", "Q4_1", "Q5_0", "Q8_0",
                "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"}
        )
        raise ValueError(
            f"Unknown quantization '{quant}' for GGUF export. "
            f"Expected one of: {', '.join(allowed)}"
        )
    return dtype, dtype.value not in {
        ggml.F16.value,
        ggml.F32.value,
    }


def _hf_to_gguf_name(name: str) -> str:
    """Map a HuggingFace/Llama tensor name to its GGUF equivalent.

    Args:
        name: Source tensor name (HF/Llama convention).

    Returns:
        GGUF-canonical name (e.g. ``token_embd.weight``,
        ``blk.0.attn_q.weight``). Unrecognised names are passed through
        unchanged.
    """
    # Global tensors.
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"
    # Per-layer tensors.
    prefix = "model.layers."
    if name.startswith(prefix):
        rest = name[len(prefix):]
        idx, _, tail = rest.partition(".")
        head = f"blk.{idx}."
        if tail == "self_attn.q_proj.weight":
            return head + "attn_q.weight"
        if tail == "self_attn.k_proj.weight":
            return head + "attn_k.weight"
        if tail == "self_attn.v_proj.weight":
            return head + "attn_v.weight"
        if tail == "self_attn.o_proj.weight":
            return head + "attn_output.weight"
        if tail == "input_layernorm.weight":
            return head + "attn_norm.weight"
        if tail == "post_attention_layernorm.weight":
            return head + "ffn_norm.weight"
        if tail == "mlp.gate_proj.weight":
            return head + "ffn_gate.weight"
        if tail == "mlp.up_proj.weight":
            return head + "ffn_up.weight"
        if tail == "mlp.down_proj.weight":
            return head + "ffn_down.weight"
    return name  # pass through


def _permute_qk(weights: "np.ndarray", n_head: int, n_head_kv: int) -> "np.ndarray":
    """Permute attention Q/K weights for the GGUF ``llama`` architecture.

    flatbuild trains with HuggingFace's ``rotate_half`` RoPE, which pairs
    head dimensions ``i`` and ``i + head_dim/2`` (half-split / NEOX). The
    GGUF ``llama`` arch tells ggml to apply NORM (consecutive-pair) RoPE,
    which expects Q/K pre-permuted to interleaved layout — exactly what
    llama.cpp's ``convert_hf_to_gguf.py`` does for the llama family.

    Args:
        weights: Q/K projection weight of shape ``(n_out, n_embd)``.
        n_head: Number of query heads.
        n_head_kv: Number of KV heads (GQA); permutation groups by the
            KV head count so grouped heads align correctly.

    Returns:
        Permuted weights (same shape) in interleaved layout.
    """
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (
        weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, weights.shape[1])
        .swapaxes(1, 2)
        .reshape(weights.shape)
    )



class GGUFExporter(Exporter):
    """Write a GGUF v3 file via ``gguf.GGUFWriter``."""

    def __init__(self, copy_tokenizer: bool = True) -> None:
        """Initialize.

        Args:
            copy_tokenizer: When ``True`` and the source checkpoint
                contains a tokenizer directory, copy it next to the GGUF.
        """
        self.copy_tokenizer = copy_tokenizer

    def _embed_bpe_tokenizer(
        self,
        writer,
        tokenizer_dir: str | Path | None,
        config: ExportConfig | None = None,
    ) -> None:
        """Embed the BPE tokenizer into the GGUF file as metadata.

        Args:
            writer: GGUF writer.
            tokenizer_dir: Source directory containing the BPE
                tokenizer files.
            config: Optional export config — provides the chat
                template to embed.
        """
        if tokenizer_dir is None:
            return
        from flatbuild.tokenizers.bpe import BPETokenizer

        try:
            tok = BPETokenizer.load(Path(tokenizer_dir))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not load BPE tokenizer for embedding: {exc}")
            return

        try:
            # The underlying ``tokenizers.Tokenizer`` exposes vocab, merges,
            # and special tokens through its public API.
            inner = tok._inner  # type: ignore[attr-defined]
            vocab = inner.get_vocab()  # dict[str, int]
            ordered_tokens: list[str] = [
                tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])
            ]
            writer.add_tokenizer_model("gpt2")
            writer.add_token_list(ordered_tokens)

            # Special-token IDs.
            writer.add_bos_token_id(tok.bos_token_id)
            writer.add_eos_token_id(tok.eos_token_id)
            writer.add_unk_token_id(tok.unk_token_id)
            if tok.pad_token_id is not None:
                writer.add_pad_token_id(tok.pad_token_id)

            # BPE merge rules. Read directly from tokenizer.json since
            # ``tokenizers`` 0.22.x does not expose merges via the public API.
            # llama.cpp expects space-separated "A B" pairs.
            num_merges = 0
            tokenizer_json = Path(tokenizer_dir) / "tokenizer.json"
            if tokenizer_json.is_file():
                try:
                    with open(tokenizer_json, encoding="utf-8") as f:
                        tj = json.load(f)
                    raw_merges = tj.get("model", {}).get("merges", [])
                    formatted = [f"{a} {b}" for a, b in raw_merges]
                    num_merges = len(formatted)
                    writer.add_token_merges(formatted)
                except Exception:  # pragma: no cover - defensive
                    pass

            # Chat template. Generate from the export config so the
            # system-prompt injection (default system when caller omits
            # a system message) is always baked in, regardless of what
            # the source tokenizer_config.json contains.
            if config is not None and config.chat_template.system is not None:
                tmpl = build_chat_template(config.chat_template)
                writer.add_string("tokenizer.chat_template", to_flatrun_jinja(tmpl))

            # Token types (1 = NORMAL, 3 = CONTROL/SPECIAL, 4 = USER_DEFINED,
            # 6 = BYTE) following the canonical GGML/llama.cpp convention.
            # llama.cpp and flatrun treat CONTROL (3) and USER_DEFINED (4)
            # as added/special tokens; marking eos/bos/unk/pad with anything
            # else (e.g. UNKNOWN=2) makes runtimes BPE-split the special
            # markers instead of emitting their single ids.
            special_ids = {
                tok.bos_token_id,
                tok.eos_token_id,
                tok.unk_token_id,
            }
            if tok.pad_token_id is not None:
                special_ids.add(tok.pad_token_id)
            token_types = [
                3 if i in special_ids else 1 for i in range(len(ordered_tokens))
            ]
            try:
                writer.add_token_types(token_types)
            except Exception:  # pragma: no cover - new gguf-python
                pass

            logger.info(
                f"Embedded BPE tokenizer ({len(ordered_tokens)} tokens, "
                f"{num_merges} merges) into GGUF."
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to embed tokenizer into GGUF: {exc}")

    def export(
        self,
        model: FlatbuildModel,
        output_dir: str | Path,
        *,
        config: ExportConfig | None = None,
        tokenizer_dir: str | Path | None = None,
    ) -> Path:
        """Write the GGUF file to ``output_dir/model.gguf``.

        The tokenizer is **always** attached when one is available —
        either via the ``tokenizer_dir`` kwarg, via ``config.tokenizer_path``,
        or omitted (then just weights are written). Files are placed at
        the same level as ``model.gguf`` so llama.cpp / flatrun can find
        them at the canonical paths.

        Args:
            model: Trained FlatbuildModel.
            output_dir: Destination directory.
            config: Optional export config (tokenizer_path may be honored).
            tokenizer_dir: Optional explicit tokenizer directory. When
                provided, takes precedence over ``config.tokenizer_path``.

        Returns:
            The output directory.
        """
        try:
            from gguf import GGUFWriter, GGMLQuantizationType
            from gguf.quants import GGML_QUANT_SIZES, quantize
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "GGUF export requires the optional dependency `gguf`. "
                "Install it with:  pip install -e '.[gguf]'"
            ) from exc

        from flatbuild.exporters._tokenizer_copy import copy_tokenizer

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "model.gguf"

        cfg = model.config
        state_dict = model.state_dict_llama()

        # Resolve quantization. Config.quant may carry the value; the caller
        # can also pass it as ExportConfig.quant. Defaults to F16.
        quant_name = getattr(config, "quant", None) or "f16"
        quant_type, is_quantized = _resolve_quant_type(quant_name, GGMLQuantizationType)

        # gguf-python only implements legacy quants (Q4_0, Q4_1, Q5_0, Q5_1,
        # Q8_0). K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K) are not yet
        # implemented. Fail fast with a clear message rather than silently
        # writing raw F32 data.
        if is_quantized:
            import numpy as np  # noqa: F401 - runtime use only in this block
            from gguf.quants import quantize as _q_check

            test_arr = np.zeros((256, 256), dtype=np.float32)
            try:
                _q_check(test_arr, quant_type)
            except NotImplementedError as exc:
                raise ValueError(
                    f"Quantization '{quant_name}' is not yet implemented in "
                    f"the installed gguf-python version. "
                    f"Supported quant types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0. "
                    f"Use --quant Q4_0 or --quant Q8_0 instead."
                ) from exc

        writer = GGUFWriter(str(out_path), arch="llama")
        # The ``arch="llama"`` constructor already inserts
        # ``general.architecture``; we add the model-specific fields here.
        writer.add_context_length(cfg.context_length)
        writer.add_embedding_length(cfg.hidden_dim)
        writer.add_feed_forward_length(cfg.ffn_dim)
        writer.add_block_count(cfg.n_layers)
        writer.add_head_count(cfg.n_heads)
        writer.add_head_count_kv(cfg.n_kv_heads or max(1, cfg.n_heads // 2))
        head_dim = int(getattr(cfg, "head_dim", cfg.hidden_dim // cfg.n_heads))
        writer.add_key_length(head_dim)
        writer.add_rope_freq_base(cfg.rope_theta)
        writer.add_layer_norm_rms_eps(1e-6)
        export_name = (
            getattr(config, "model_name", None)
            or getattr(model.config, "name", None)
            or "flatbuild"
        )
        writer.add_name(export_name)

        # ---- Identity / provenance metadata (params, publisher, quant). ----
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"GGUF export: {num_params:,} params, dtype={quant_type.name}, "
            f"quantized={is_quantized}"
        )
        # ``total.parameters`` is the widely-used GGUF/HF key for param count.
        writer.add_uint64("total.parameters", num_params)
        # Publisher / organisation metadata.
        publisher = getattr(config, "publisher", None) or "flatbuild"
        if publisher:
            writer.add_string("general.publisher", publisher)
        # Register a clear per-file quant label so llama.cpp reports it.
        writer.add_uint32("general.quantized", int(is_quantized))
        # flatbuild uses half-split (NEOX) RoPE format internally.
        # Export to GGUF without permutation - flatrun reads rope_interleaved from metadata.
        writer.add_string("general.rope_format", "half-split")

        # Tensor transport: flatten weights and, when quantizing, call
        # ``gguf.quants.quantize()`` explicitly (the ``raw_dtype`` parameter
        # on add_tensor is only metadata — it does not quantize the data).
        # Like llama.cpp, keep a tensor on F16 when its last axis isn't a
        # multiple of the quant block size (e.g. vocab embeddings with
        # non-block-aligned sizes, norm tensors with odd shapes).
        fallback_dtype = GGMLQuantizationType.F16
        block_size = GGML_QUANT_SIZES[quant_type][0] if is_quantized else 0
        quantized_count = 0
        fallback_count = 0
        for name, tensor in state_dict.items():
            flat_t = tensor.detach().contiguous().cpu()
            arr = flat_t.numpy().astype("float32", copy=False)
            gguf_name = _hf_to_gguf_name(name)
            # NOTE: flatbuild uses half-split (NEOX) RoPE internally.
            # No permutation applied here - weights stay in HF format.
            last = arr.shape[-1]
            if is_quantized and last % block_size == 0:
                arr = quantize(arr, quant_type)
                writer.add_tensor(gguf_name, arr, raw_dtype=quant_type)
                quantized_count += 1
            elif is_quantized:
                writer.add_tensor(gguf_name, arr, raw_dtype=fallback_dtype)
                fallback_count += 1
            else:
                writer.add_tensor(gguf_name, arr)
        if is_quantized:
            logger.info(
                f"Quantized {quantized_count} tensors, "
                f"{fallback_count} fallback F16 (block_size={block_size})."
            )

        # Embed the BPE tokenizer (vocab + merges + special tokens)
        # directly into the GGUF file so the export is fully
        # self-contained. ``llama.cpp`` and ``flatrun`` read this
        # block when the model is loaded — no sidecar file needed.
        # Resolve source the same way as the sidecar copy below.
        tokenizer_source = tokenizer_dir
        if tokenizer_source is None and config is not None:
            tokenizer_source = getattr(config, "tokenizer_path", None)
        self._embed_bpe_tokenizer(writer, tokenizer_source, config)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        logger.info(f"Wrote {out_path}")

        if self.copy_tokenizer:
            src = tokenizer_dir
            if src is None and config is not None:
                src = getattr(config, "tokenizer_path", None)
            if src is not None:
                copy_tokenizer(src, output_dir)

        # Companion config.json so flatrun / transformers tools can sanity-check.
        import json

        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "format": "gguf",
                    "gguf_path": str(out_path.name),
                    "model_type": "llama",
                    "hidden_size": cfg.hidden_dim,
                    "num_attention_heads": cfg.n_heads,
                    "num_hidden_layers": cfg.n_layers,
                },
                f,
                indent=2,
            )

        del state_dict
        import gc

        gc.collect()
        return output_dir
