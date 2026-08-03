"""Export a Flatbuild model to SafeTensors with an HF-compatible config.json.

Tokenizer artifacts (``tokenizer.json`` etc.) are written **flat** in
the same directory as ``model.safetensors`` — matching HuggingFace
convention so a single ``from_pretrained(path)`` Just Works.
"""

from __future__ import annotations

import json
from pathlib import Path

from flatbuild.config import ExportConfig
from flatbuild.exporters._tokenizer_copy import copy_tokenizer
from flatbuild.exporters.base import Exporter
from flatbuild.models import FlatbuildModel
from flatbuild.tokenizers import build_chat_template
from flatbuild.tokenizers.template import to_flatrun_jinja
from flatbuild.utils import get_logger

logger = get_logger(__name__)


class SafeTensorsExporter(Exporter):
    """Write a single ``model.safetensors`` plus Llama-style ``config.json``.

    The resulting directory can be loaded directly by ``flatrun`` and the
    HuggingFace Transformers library.
    """

    def __init__(self, copy_tokenizer: bool = True) -> None:
        """Initialize.

        Args:
            copy_tokenizer: When ``True`` and a tokenizer is available
                (either via ``config.tokenizer_path`` or via the
                optional ``tokenizer`` kwarg on ``export()``), copy its
                files flat alongside the model.
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
        """Write SafeTensors + config + tokenizer files to ``output_dir``.

        Args:
            model: Trained model.
            output_dir: Destination directory.
            config: Optional export config (tokenizer_path may be honored).
            tokenizer_dir: Optional explicit tokenizer directory. When
                provided, it takes precedence over ``config.tokenizer_path``.

        Returns:
            The output directory.
        """
        from safetensors.torch import save_file as _save_file

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        state_dict = model.state_dict_llama()
        _save_file(state_dict, str(output_dir / "model.safetensors"))
        logger.info(f"Wrote {output_dir / 'model.safetensors'}")

        cfg = model.config
        config_json = {
            "architectures": ["FlatbuildModel"],
            "model_type": "flatbuild",
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
            "transformers_version": "4.40.0",
        }
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_json, f, indent=2)

        if self.copy_tokenizer:
            self._attach_tokenizer(output_dir, config, tokenizer_dir)

        return output_dir

    def _attach_tokenizer(
        self,
        output_dir: Path,
        config: ExportConfig | None,
        tokenizer_dir: str | Path | None,
    ) -> None:
        """Resolve a tokenizer directory from explicit kwarg or config.

        Args:
            output_dir: Where to land the tokenizer files (root level).
            config: Optional export config — may carry ``tokenizer_path``.
            tokenizer_dir: Direct override for the tokenizer source.
        """
        src = tokenizer_dir
        if src is None and config is not None:
            src = getattr(config, "tokenizer_path", None)
        if src is None:
            return
        copy_tokenizer(src, output_dir)
        # Patch tokenizer_config.json so the chat_template carries
        # the system-prompt injection (default system when caller
        # omits a system message).  This mirrors the GGUF exporter
        # behaviour and keeps safetensors + GGUF consistent.
        if config is not None and config.chat_template.system is not None:
            tc_path = output_dir / "tokenizer_config.json"
            if tc_path.is_file():
                try:
                    with open(tc_path, encoding="utf-8") as f:
                        tc = json.load(f)
                    tmpl = build_chat_template(config.chat_template)
                    tc["chat_template"] = to_flatrun_jinja(tmpl)
                    with open(tc_path, "w", encoding="utf-8") as f:
                        json.dump(tc, f, indent=2)
                except Exception:  # pragma: no cover - defensive
                    pass
