"""Checkpoint save / load.

A Flatbuild checkpoint directory contains:

- ``config.yaml`` — the YAML config used for the run (for reproducibility).
- ``model_state_dict.pt`` — full torch state dict (incl. optimizer & rng).
- ``model.safetensors`` — Llama-compatible weight tensors (single shard).
- ``tokenizer/`` — the trained tokenizer, ready for inference.
- ``trainer_state.json`` — global step, epoch, last loss, RNG seed.
- ``metrics.jsonl`` — JSONL log of training metrics.
- ``history.json`` — list of step metrics (loss / val_loss / etc.).

The manager keeps only the last N checkpoints and exposes
``load_latest()`` for resuming.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from flatbuild.config import FlatBuildConfig
from flatbuild.utils import get_logger, write_json

logger = get_logger(__name__)


@dataclass
class CheckpointState:
    """Serializable runtime state saved alongside model weights."""

    global_step: int = 0
    epoch_index: int = 0
    last_loss: float | None = None
    best_val_loss: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "epoch_index": self.epoch_index,
            "last_loss": self.last_loss,
            "best_val_loss": self.best_val_loss,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointState":
        extra = {k: v for k, v in data.items() if k not in {"global_step", "epoch_index", "last_loss", "best_val_loss"}}
        return cls(
            global_step=int(data.get("global_step", 0)),
            epoch_index=int(data.get("epoch_index", 0)),
            last_loss=data.get("last_loss"),
            best_val_loss=data.get("best_val_loss"),
            extra=extra,
        )


class CheckpointManager:
    """Manage checkpoint versions inside a single run folder.

    Args:
        run_dir: Run folder (``outputs/<name>/<timestamp>``).
        max_to_keep: How many latest step checkpoints to retain (default 3).
    """

    def __init__(self, run_dir: Path, max_to_keep: int = 3) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self._step_dirs: list[Path] = []

    @property
    def final_dir(self) -> Path:
        """Location of the final consolidated checkpoint."""
        return self.checkpoints_dir / "final"

    @property
    def latest_dir(self) -> Path | None:
        """Return the path of the most recent step checkpoint, if any."""
        steps = sorted(self._step_dirs, key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
        return steps[-1] if steps else None

    def save_step(
        self,
        *,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        tokenizer_dir: Path | None,
        config: FlatBuildConfig,
        state: CheckpointState,
        dataset_path: str | None = None,
    ) -> Path:
        """Save a numbered step checkpoint.

        Returns:
            Path to the freshly-saved checkpoint directory.
        """
        step_dir = self.checkpoints_dir / f"step-{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # State dict (Llama keys for cross-ecosystem interop).
        sd = model.state_dict_llama()
        torch.save(sd, step_dir / "model_state_dict.pt")
        from safetensors.torch import save_file as _save_safetensors

        _save_safetensors(sd, str(step_dir / "model.safetensors"))

        # Optimizer + RNG (for exact resume — optional but useful).
        if optimizer is not None:
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                step_dir / "optimizer.pt",
            )

        # Tokenizer copy + config snapshot + trainer state.
        if tokenizer_dir is not None and Path(tokenizer_dir).exists():
            target = step_dir / "tokenizer"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(Path(tokenizer_dir), target)
        config.to_yaml(step_dir / "config.yaml")
        state_dict = state.to_dict()
        if dataset_path:
            state_dict["dataset_path"] = dataset_path
        write_json(step_dir / "trainer_state.json", state_dict)

        self._step_dirs.append(step_dir)
        if len(self._step_dirs) > self.max_to_keep:
            old = self._step_dirs.pop(0)
            shutil.rmtree(old, ignore_errors=True)

        return step_dir

    def save_final(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        tokenizer_dir: Path | None,
        config: FlatBuildConfig,
        state: CheckpointState,
        dataset_path: str | None = None,
    ) -> Path:
        """Save the ``final/`` checkpoint (always retained)."""
        self.final_dir.mkdir(parents=True, exist_ok=True)
        sd = model.state_dict_llama()
        torch.save(sd, self.final_dir / "model_state_dict.pt")
        from safetensors.torch import save_file as _save_safetensors

        _save_safetensors(sd, str(self.final_dir / "model.safetensors"))
        if optimizer is not None:
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                self.final_dir / "optimizer.pt",
            )
        if tokenizer_dir is not None and Path(tokenizer_dir).exists():
            target = self.final_dir / "tokenizer"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(Path(tokenizer_dir), target)
        config.to_yaml(self.final_dir / "config.yaml")
        state_dict = state.to_dict()
        if dataset_path:
            state_dict["dataset_path"] = dataset_path
        write_json(self.final_dir / "trainer_state.json", state_dict)
        # Refresh ``history.json`` with the latest metrics view.
        history_path = self.run_dir / "history.json"
        if history_path.exists():
            shutil.copyfile(history_path, self.final_dir / "history.json")
        return self.final_dir

    @staticmethod
    def load(directory: str | Path) -> dict[str, Any]:
        """Load a checkpoint directory into a dict.

        Returns:
            Dictionary with ``model_state_dict``, optional ``optimizer``,
            optional ``tokenizer_dir``, ``config`` (parsed :class:`FlatBuildConfig`)
            and ``state`` (:class:`CheckpointState`).
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {directory}")

        # Prefer the Llama-format SafeTensors for cross-ecosystem use.
        safetensors_path = directory / "model.safetensors"
        pt_path = directory / "model_state_dict.pt"
        if safetensors_path.exists():
            from safetensors.torch import load_file as _load_file

            sd = _load_file(str(safetensors_path))
        else:
            sd = torch.load(pt_path, map_location="cpu", weights_only=True)

        optimizer_state: dict[str, Any] | None = None
        opt_path = directory / "optimizer.pt"
        if opt_path.exists():
            optimizer_state = torch.load(opt_path, map_location="cpu", weights_only=False)
            torch.set_rng_state(optimizer_state.get("rng_torch", torch.get_rng_state()))
            cuda_state = optimizer_state.get("rng_cuda")
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)

        cfg_path = directory / "config.yaml"
        config = FlatBuildConfig.from_yaml(cfg_path) if cfg_path.exists() else None

        state_path = directory / "trainer_state.json"
        if state_path.exists():
            with open(state_path, encoding="utf-8") as f:
                state = CheckpointState.from_dict(json.load(f))
        else:
            state = CheckpointState()

        tok_dir = directory / "tokenizer"
        # Get dataset_path from state.extra if present
        dataset_path = state.extra.get("dataset_path") if state.extra else None
        return {
            "model_state_dict": sd,
            "optimizer_state": optimizer_state,
            "tokenizer_dir": tok_dir if tok_dir.exists() else None,
            "config": config,
            "state": state,
            "directory": directory,
            "dataset_path": dataset_path,
        }


# Convenience wrappers


def save_checkpoint(directory: str | Path, **kwargs) -> Path:
    """Convenience wrapper around :class:`CheckpointManager.save_final`."""
    mgr = CheckpointManager(Path(directory).parent)
    return mgr.save_final(**kwargs)


def load_checkpoint(directory: str | Path) -> dict[str, Any]:
    """Convenience wrapper around :class:`CheckpointManager.load`."""
    return CheckpointManager.load(directory)
