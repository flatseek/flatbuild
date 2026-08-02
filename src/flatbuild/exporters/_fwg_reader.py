"""Read a Flatweight ``.fwg`` archive and reconstruct a PyTorch state dict.

Flatweight stores tensors in 64x64 (default) tiles, each page addressable
by a deterministic hash. ``WeightFS.read_tensor(name)`` reassembles them
into a numpy array. We wrap that in a small adapter that returns a
PyTorch state dict with Llama-style keys, identical to the format
:class:`FlatbuildModel.load_state_dict_llama` understands.

Useful for loading weights produced by :class:`flatbuild.exporters.fwg.FWGExporter`
or any other tool that writes ``.fwg`` archives (notably flatweight's own
CLI: ``flatweight convert`` / ``flatweight build``).
"""

from __future__ import annotations

from pathlib import Path

from flatbuild.utils import get_logger

logger = get_logger(__name__)


def load_fwg_state_dict(
    fwg_path: str | Path,
) -> tuple[dict, dict]:
    """Load a ``.fwg`` archive and return ``(state_dict, metadata)``.

    Args:
        fwg_path: Path to a ``.fwg`` archive or Directory Mode WeightFS.

    Returns:
        Tuple ``(state_dict, metadata)`` where ``state_dict`` maps
        ``tensor_name -> torch.Tensor`` and ``metadata`` is a
        dictionary containing ``tile_size``, ``tensor_count``, etc.

    Raises:
        ImportError: If ``flatweight`` is not installed.
        FileNotFoundError: If the archive does not exist.
    """
    try:
        import flatweight as fw
        from flatweight.weightfs import WeightFS  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "FWG reading requires the optional sibling package `flatweight`. "
            "Install it with:  pip install -e '.[fwg]'"
        ) from exc

    import numpy as np
    import torch

    fwg_path = Path(fwg_path)
    if not fwg_path.exists():
        raise FileNotFoundError(f"FWG archive not found: {fwg_path}")

    logger.info(f"Opening WeightFS at {fwg_path}")
    wf = WeightFS(fwg_path)
    manifest = wf.manifest
    # Sync tile_size from the manifest. The constructor default may
    # not match the tile_size the writer actually used, in which case
    # ``read_tensor`` will compute wrong page-grid dimensions.
    manifest_tile_size = getattr(manifest, "tile_size", None)
    if isinstance(manifest_tile_size, int) and manifest_tile_size > 0:
        wf.tile_size = manifest_tile_size

    state_dict: dict[str, torch.Tensor] = {}
    for tensor_name, info in manifest.tensors.items():
        arr = wf.read_tensor(tensor_name)
        # ``read_tensor`` already returns a numpy array with the right
        # shape and dtype. Wrap it into a writable copy so
        # ``torch.from_numpy`` is happy and the result owns its memory.
        np_arr = np.ascontiguousarray(arr).copy()
        # Cast byte widths to torch dtypes.
        if np_arr.dtype == np.float32:
            torch_dtype = torch.float32
        elif np_arr.dtype == np.float16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32  # fallback
        tensor = torch.from_numpy(np_arr).to(torch_dtype)
        state_dict[tensor_name] = tensor

    metadata = {
        "tile_size": getattr(manifest, "tile_size", wf.tile_size),
        "tensor_count": manifest.tensor_count,
        "page_count": manifest.page_count,
        "storage_mode": wf.storage_mode.value if hasattr(wf.storage_mode, "value") else str(wf.storage_mode),
    }
    wf.close()
    logger.info(
        f"Loaded {len(state_dict)} tensors ({metadata['tensor_count']} unique) "
        f"from {fwg_path} ({metadata['storage_mode']} mode)"
    )
    return state_dict, metadata


__all__ = ["load_fwg_state_dict"]
