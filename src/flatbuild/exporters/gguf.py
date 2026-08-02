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

from pathlib import Path

from flatbuild.config import ExportConfig
from flatbuild.exporters.base import Exporter
from flatbuild.models import FlatbuildModel
from flatbuild.utils import get_logger

logger = get_logger(__name__)


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
    ) -> None:
        """Embed the BPE tokenizer into the GGUF file as metadata.

        Args:
            writer: An open :class:`gguf.GGUFWriter`.
            tokenizer_dir: Source directory containing the BPE
                ``tokenizer.json`` saved by :class:`BPETokenizer.save`.
                When ``None`` this is a no-op (flatrun will fall back
                to the sidecar ``tokenizer/`` directory or the in-vocab
                special tokens).
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

            # BPE merge rules. ``inner.model.get_merges()`` returns
            # ``Merge(a, b)`` namedtuples; llama.cpp expects
            # space-separated "A B" pairs.
            try:
                merges = inner.model.get_merges()
                formatted = [f"{m.a} {m.b}" for m in merges]
                writer.add_token_merges(formatted)
            except Exception:  # pragma: no cover - non-BPE backend
                pass

            # Token types (1 = NORMAL, 2 = CONTROL/CONTROLLED, 3 = BYTE).
            # Mark all known special tokens as CONTROL.
            special_ids = {
                tok.bos_token_id,
                tok.eos_token_id,
                tok.unk_token_id,
            }
            if tok.pad_token_id is not None:
                special_ids.add(tok.pad_token_id)
            token_types = [
                2 if i in special_ids else 1 for i in range(len(ordered_tokens))
            ]
            try:
                writer.add_token_types(token_types)
            except Exception:  # pragma: no cover - new gguf-python
                pass

            logger.info(
                f"Embedded BPE tokenizer ({len(ordered_tokens)} tokens, "
                f"{len(formatted) if 'formatted' in locals() else 0} merges) into GGUF."
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
            from gguf import GGUFWriter
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

        writer = GGUFWriter(str(out_path), arch="llama")
        # The ``arch="llama"`` constructor already inserts
        # ``general.architecture`` and ``general.file_type``; we
        # only need to add the model-specific fields here.
        writer.add_context_length(cfg.context_length)
        writer.add_embedding_length(cfg.hidden_dim)
        writer.add_feed_forward_length(cfg.ffn_dim)
        writer.add_block_count(cfg.n_layers)
        writer.add_head_count(cfg.n_heads)
        writer.add_head_count_kv(cfg.n_kv_heads or max(1, cfg.n_heads // 2))
        writer.add_rope_freq_base(cfg.rope_theta)
        writer.add_layer_norm_rms_eps(1e-6)
        writer.add_name(getattr(model.config, "name", "flatbuild") or "flatbuild")

        # Flatten linear weight matrices so llama.cpp's tensor layout
        # matches LlamaForCausalLM (no transpose needed downstream).
        for name, tensor in state_dict.items():
            flat_t = tensor.detach().contiguous().cpu()
            arr = flat_t.numpy().astype("float32", copy=False)
            gguf_name = _hf_to_gguf_name(name)
            writer.add_tensor(gguf_name, arr)

        # Embed the BPE tokenizer (vocab + merges + special tokens)
        # directly into the GGUF file so the export is fully
        # self-contained. ``llama.cpp`` and ``flatrun`` read this
        # block when the model is loaded — no sidecar file needed.
        # Resolve source the same way as the sidecar copy below.
        tokenizer_source = tokenizer_dir
        if tokenizer_source is None and config is not None:
            tokenizer_source = getattr(config, "tokenizer_path", None)
        self._embed_bpe_tokenizer(writer, tokenizer_source)

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
