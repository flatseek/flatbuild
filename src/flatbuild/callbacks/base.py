"""Callback base classes and hook definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flatbuild.trainer.trainer import FlatbuildTrainer


@dataclass
class CallbackContext:
    """Context passed to callbacks on each hook."""

    trainer: "FlatbuildTrainer"
    state: dict[str, Any]


class TrainerHooks:
    """String names of the trainer hooks callbacks can subscribe to."""

    TRAIN_BEGIN = "train_begin"
    TRAIN_END = "train_end"
    EPOCH_BEGIN = "epoch_begin"
    EPOCH_END = "epoch_end"
    STEP_BEGIN = "step_begin"
    STEP_END = "step_end"
    EVAL_BEGIN = "eval_begin"
    EVAL_END = "eval_end"
    EARLY_STOP = "early_stop"
    CHECKPOINT_SAVED = "checkpoint_saved"


class Callback:
    """Lightweight callback base.

    Subclasses override the hooks they care about. Hooks have no-op
    defaults so the trainer can call them unconditionally.
    """

    def on_train_begin(self, ctx: CallbackContext) -> None:
        """Hook called once at train start."""

    def on_train_end(self, ctx: CallbackContext) -> None:
        """Hook called once at train end."""

    def on_epoch_begin(self, ctx: CallbackContext) -> None:
        """Hook called at epoch start."""

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        """Hook called at epoch end."""

    def on_step_begin(self, ctx: CallbackContext) -> None:
        """Hook called at step start."""

    def on_step_end(self, ctx: CallbackContext) -> None:
        """Hook called at step end."""

    def on_eval_begin(self, ctx: CallbackContext) -> None:
        """Hook called before each evaluation pass."""

    def on_eval_end(self, ctx: CallbackContext) -> None:
        """Hook called after each evaluation pass."""

    def on_early_stop(self, ctx: CallbackContext) -> None:
        """Hook called once when early stopping triggers."""

    def on_checkpoint_saved(self, ctx: CallbackContext) -> None:
        """Hook called after every checkpoint save."""


def collect_callbacks(items) -> list[Callback]:
    """Filter an iterable down to ``Callback`` instances."""
    return [c for c in items if isinstance(c, Callback)]
