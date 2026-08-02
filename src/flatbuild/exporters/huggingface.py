"""Export a Flatbuild model to a HuggingFace Transformers-compatible directory.

The output layout matches a LlamaForCausalLM directory exactly:

- ``config.json``
- ``generation_config.json``
- ``model.safetensors``
- ``tokenizer.json`` (flat, when available)
- ``tokenizer_config.json`` (flat, when available)

A user can load the directory with::

    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoModelForCausalLM.from_pretrained("path/to/export")
    AutoTokenizer.from_pretrained("path/to/export")

The exporter does not require the ``transformers`` package at runtime —
it only writes JSON + SafeTensors files that match the format
HuggingFace expects.
"""

from __future__ import annotations

import json
from pathlib import Path

from flatbuild.config import ExportConfig
from flatbuild.exporters._tokenizer_copy import copy_tokenizer
from flatbuild.exporters.base import Exporter
from flatbuild.models import FlatbuildModel
from flatbuild.utils import get_logger

logger = get_logger(__name__)


class HuggingFaceExporter(Exporter):
    """Write a HuggingFace-compatible model directory."""

    def __init__(self, copy_tokenizer: bool = True) -> None:
        """Initialize.

        Args:
            copy_tokenizer: When ``True``, attach tokenizer files flat at
                the export root (next to ``model.safetensors``).
        """
        self.copy_tokenizer = copy_tokenizer

    def export(
        self,
        model: FlatbuildModel,
        output_dir: str | Path,
        *,
        config: ExportConfig | None = None,
        tokenizer_dir: str | Path | None = None,
    ) -> Path:
        """Write the model and config in HF format.

        Args:
            model: Trained model.
            output_dir: Destination directory.
            config: Optional export config (tokenizer_path may be honored).
            tokenizer_dir: Optional explicit tokenizer directory. When
                provided, takes precedence over ``config.tokenizer_path``.

        Returns:
            The output directory.
        """
        from safetensors.torch import save_file as _save_file

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        state_dict = model.state_dict_llama()
        _save_file(state_dict, str(output_dir / "model.safetensors"))

        cfg = model.config
        config_json = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_act": "silu",
            "hidden_size": cfg.hidden_dim,
            "intermediate_size": cfg.ffn_dim,
            "num_attention_heads": cfg.n_heads,
            "num_hidden_layers": cfg.n_layers,
            "num_key_value_heads": cfg.n_kv_heads or max(1, cfg.n_heads // 2),
            "max_position_embeddings": cfg.context_length,
            "rope_theta": cfg.rope_theta,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": cfg.tie_embeddings,
            "vocab_size": cfg.vocab_size,
            "torch_dtype": "float32",
            "use_cache": True,
            "transformers_version": "4.40.0",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        }
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_json, f, indent=2)

        generation_config = {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "transformers_version": "4.40.0",
        }
        with open(output_dir / "generation_config.json", "w", encoding="utf-8") as f:
            json.dump(generation_config, f, indent=2)

        if self.copy_tokenizer:
            src = tokenizer_dir
            if src is None and config is not None:
                src = getattr(config, "tokenizer_path", None)
            if src is not None:
                copy_tokenizer(src, output_dir)

        return output_dir
