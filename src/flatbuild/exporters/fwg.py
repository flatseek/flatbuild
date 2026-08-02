"""Export a Flatbuild model to the Flatweight (.fwg) format.

Requires the sibling ``flatweight`` package::

    pip install -e ".[fwg]"

Strategy:
1. Write Llama-keyed tensors to a temp directory as a single
   ``model.safetensors`` plus a sibling ``config.json``.
2. Call :func:`flatweight.converter.convert` (the generic Safetensors
   path) — this preserves our Llama-keyed tensor names, so the
   round-trip stays 1:1 with Flatbuild's own state dict. The
   HF-aware converter ``convert_hf_safetensors`` would rename tensors
   to GGUF-style names and transpose linear matrices, which is
   undesirable here.
3. Choose a tile_size ≥ the largest tensor dimension so that every
   2-D tensor fits in a single page. This matches the shape of the
   real ``bons.fwg`` / ``smoll.fwg`` / ``qq.fwg`` artifacts (where
   every tensor is a single page because they are quantized), and is
   also natively consumable by Flatweight's C++/llama.cpp packed
   loader — which has known issues reconstructing multi-page 2-D
   tiles from binary-index order. For our compact conversational
   models the largest tensor dimension is well under the page
   bookkeeping budget, so 1-page-per-tensor is the right trade-off.
4. Manually enrich ``WeightFSManifest.metadata`` with the model's
   architecture so consumers that look at the manifest (Flatweight
   Runtime, ``flatweight inspect``) see arch + dimensions.
5. Pack the directory into a single ``.fwg``.

The result is consumable by ``flatweight inspect /path/to/checkpoint.fwg``,
``flatweight.WeightFS(path)``, and the native llama.cpp flatweight loader.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from flatbuild.config import ExportConfig
from flatbuild.exporters.base import Exporter
from flatbuild.models import FlatbuildModel
from flatbuild.utils import get_logger

logger = get_logger(__name__)


def _max_tensor_dim(state_dict: dict) -> int:
    """Return the largest dimension across all 2-D tensors.

    Args:
        state_dict: Mapping of tensor name → tensor.

    Returns:
        Maximum dim (≥ 1). Used to choose a tile_size that yields
        1 page per 2-D tensor.
    """
    best = 1
    for tensor in state_dict.values():
        shape = tensor.shape
        if len(shape) >= 2:
            best = max(best, int(shape[0]), int(shape[1]))
        elif len(shape) == 1:
            best = max(best, int(shape[0]))
    return best


class FWGExporter(Exporter):
    """Write a Flatweight ``.fwg`` archive (1 page per tensor)."""

    def __init__(self, tile_size: int | None = None, copy_tokenizer: bool = True) -> None:
        """Initialize.

        Args:
            tile_size: Flatweight tile size in elements per side.
                ``None`` (default) picks a value ≥ ``max(tensor_dim)``
                so every 2-D tensor is one page (recommended for
                interop with Flatweight's native loader). Pass an
                explicit integer to override.
            copy_tokenizer: When ``True`` and a tokenizer is available,
                copy it next to the .fwg output for convenience.
        """
        self.tile_size = tile_size
        self.copy_tokenizer = copy_tokenizer

    def export(
        self,
        model: FlatbuildModel,
        output_dir: str | Path,
        *,
        config: ExportConfig | None = None,
        tokenizer_dir: str | Path | None = None,
    ) -> Path:
        """Write the .fwg archive to ``output_dir/model.fwg``.

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
            from flatweight.converter import convert, pack_weightfs
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "FWG export requires the optional sibling package `flatweight`. "
                "Install it with:  pip install -e '.[fwg]'"
            ) from exc

        from safetensors.torch import save_file as _save_file

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = model.config
        state_dict = model.state_dict_llama()

        # Choose tile_size so every 2-D tensor is a single page.
        tile = int(self.tile_size) if self.tile_size is not None else _max_tensor_dim(state_dict)
        # Round up to a power of two (a sensible default convention).
        tile = max(1, 1 << max(0, (tile - 1).bit_length()))

        # Save state into a temp directory in the format flatweight expects.
        with tempfile.TemporaryDirectory(prefix="flatbuild_fwg_") as tmp:
            tmp_path = Path(tmp)
            work = tmp_path / "weights"
            work.mkdir(parents=True, exist_ok=True)

            # 1. Drop model.safetensors with Llama-keyed tensors.
            _save_file(state_dict, str(work / "model.safetensors"))

            # 2. Drop a config.json — flatweight's generic ``convert``
            # does not require it, but downstream consumers do.
            with open(work / "config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_type": "llama",
                        "architectures": ["LlamaForCausalLM"],
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
                    },
                    f,
                    indent=2,
                )

            # 3. Convert → directory WeightFS.
            weightfs_dir = tmp_path / "weightfs"
            weightfs_dir.mkdir(parents=True, exist_ok=True)
            manifest = convert(str(work), str(weightfs_dir), tile_size=tile)

            # 4. Enrich the manifest with architecture metadata that
            # flatweight's generic ``convert`` leaves empty.
            n_kv = cfg.n_kv_heads or max(1, cfg.n_heads // 2)
            try:
                # Some flatweight versions expose ``metadata`` as a dict attribute.
                from flatweight.core import WeightFSManifest  # type: ignore[attr-defined]

                if isinstance(manifest.metadata, dict):
                    manifest.metadata.update(
                        {
                            "architecture": "llama",
                            "model_name": "flatbuild",
                            "context_length": cfg.context_length,
                            "embedding_length": cfg.hidden_dim,
                            "block_count": cfg.n_layers,
                            "feed_forward_length": cfg.ffn_dim,
                            "num_attention_heads": cfg.n_heads,
                            "num_kv_heads": n_kv,
                            "rms_norm_eps": 1e-6,
                            "rope_freq_base": cfg.rope_theta,
                            "vocab_size": cfg.vocab_size,
                        }
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Could not enrich manifest metadata: {exc}")

            # 5. Persist the enriched manifest JSON next to the WeightFS
            # (so even ``PackedStorage.manifest`` falls back to it).
            try:
                with open(weightfs_dir / "manifest.json", "r", encoding="utf-8") as f:
                    man_data = json.load(f)
            except FileNotFoundError:
                man_data = {}
            man_data.setdefault("metadata", {}).update(
                {
                    "architecture": "llama",
                    "model_name": "flatbuild",
                    "context_length": cfg.context_length,
                    "embedding_length": cfg.hidden_dim,
                    "block_count": cfg.n_layers,
                    "feed_forward_length": cfg.ffn_dim,
                    "num_attention_heads": cfg.n_heads,
                    "num_kv_heads": n_kv,
                    "rms_norm_eps": 1e-6,
                    "rope_freq_base": cfg.rope_theta,
                    "vocab_size": cfg.vocab_size,
                }
            )
            # Forward-compat: the source WeightFS directory has
            # ``storage_mode: "directory"`` written by ``convert``.
            # Promote it to ``"packed"`` since we're about to write
            # a single ``.fwg`` archive. ``PackedStorage.manifest``
            # already overrides this in memory for current readers,
            # but a future Flatrun backend that consults the field
            # would otherwise get the wrong answer.
            man_data["storage_mode"] = "packed"
            with open(weightfs_dir / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(man_data, f, indent=2)

            # 6. Pack into a single .fwg.
            target_fwg = output_dir / "model.fwg"
            pack_weightfs(str(weightfs_dir), str(target_fwg), tile_size=tile)
            logger.info(
                f"Wrote {target_fwg} ({target_fwg.stat().st_size:,} bytes, "
                f"{manifest.tensor_count} tensors / {manifest.page_count} pages, tile_size={tile})"
            )

        # Sidecar config so flatrun / flatweight tools can identify the file.
        with open(output_dir / "fwg_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "format": "fwg",
                    "model_type": "flatbuild",
                    "fwg_path": "model.fwg",
                    "tile_size": tile,
                    "model": {
                        "hidden_size": cfg.hidden_dim,
                        "num_attention_heads": cfg.n_heads,
                        "num_hidden_layers": cfg.n_layers,
                        "num_key_value_heads": n_kv,
                    },
                },
                f,
                indent=2,
            )

        if self.copy_tokenizer:
            from flatbuild.exporters._tokenizer_copy import copy_tokenizer

            src = tokenizer_dir
            if src is None and config is not None:
                src = getattr(config, "tokenizer_path", None)
            if src is not None:
                copy_tokenizer(src, output_dir)

        return output_dir
