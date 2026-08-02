"""Logger callback: structured training / eval logging."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from flatbuild.callbacks.base import Callback, CallbackContext
from flatbuild.utils import get_logger

logger = get_logger(__name__)


class LoggerCallback(Callback):
    """Log training / eval metrics to stdout + a JSONL file.

    Args:
        log_path: Path to the JSONL file. ``None`` disables file logging.
        fields: Iterable of metric names to record; default is a useful
            default set used by the trainer.
    """

    def __init__(
        self,
        log_path: Path | None = None,
        fields: Iterable[str] | None = None,
    ) -> None:
        """Initialize the logger.

        Args:
            log_path: Where to append the structured logs.
            fields: Field names kept in the JSONL output.
        """
        self.log_path = log_path
        self.fields = list(
            fields
            or (
                "step",
                "epoch",
                "loss",
                "lr",
                "val_loss",
                "val_perplexity",
                "val_accuracy",
            )
        )
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Start fresh so re-runs don't append to stale logs.
            open(self.log_path, "w", encoding="utf-8").close()

    def on_train_begin(self, ctx: CallbackContext) -> None:
        """Print a banner at training start."""
        cfg = ctx.trainer.config
        logger.info("=" * 60)
        logger.info("Flatbuild training starting…")
        logger.info("=" * 60)
        logger.info(f"project={cfg.name!r}  vocab={cfg.model.vocab_size}  layers={cfg.model.n_layers}")
        logger.info(f"hidden={cfg.model.hidden_dim}  heads={cfg.model.n_heads}  kv_heads={cfg.model.n_kv_heads}")
        logger.info(f"batch_size={cfg.trainer.batch_size}  grad_accum={cfg.trainer.gradient_accumulation}  precision={cfg.trainer.precision.value}")

    def on_step_end(self, ctx: CallbackContext) -> None:
        """Record the periodic step log.

        Args:
            ctx: Trainer context with ``trainer.metrics`` populated.
        """
        step = ctx.trainer.global_step
        if step % max(1, ctx.trainer.config.trainer.log_every_n_steps) != 0:
            return
        rec = self._gather(ctx, include_val=False)
        line = json.dumps(rec, indent=None)
        logger.info(f"[step {rec['step']}] loss={rec.get('loss', 0):.4f} lr={rec.get('lr', 0):.2e}")
        self._append(line)

    def on_eval_end(self, ctx: CallbackContext) -> None:
        """Record an evaluation pass.

        Args:
            ctx: Trainer context with ``val_loss`` populated.
        """
        rec = self._gather(ctx, include_val=True)
        line = json.dumps(rec, indent=None)
        logger.info(
            f"[step {rec['step']}] val_loss={rec.get('val_loss', 0):.4f} "
            f"perplexity={rec.get('val_perplexity', 0):.2f} "
            f"accuracy={rec.get('val_accuracy', 0):.3f}"
        )
        self._append(line)

    def on_train_end(self, ctx: CallbackContext) -> None:
        """Print a closing banner."""
        logger.info("=" * 60)
        logger.info("Flatbuild training complete.")
        logger.info("=" * 60)

    # ------------------------------------------------------------------

    def _gather(self, ctx: CallbackContext, *, include_val: bool) -> dict:
        trainer = ctx.trainer
        state = ctx.state
        rec: dict[str, float] = {}
        if "step" in self.fields:
            rec["step"] = trainer.global_step
        if "epoch" in self.fields:
            rec["epoch"] = trainer.epoch_index
        if include_val:
            for f in ("val_loss", "val_perplexity", "val_accuracy"):
                if f in self.fields and f in state:
                    rec[f] = float(state[f])
        else:
            if "loss" in self.fields:
                rec["loss"] = float(state.get("loss", trainer.last_loss or 0.0))
            if "lr" in self.fields:
                rec["lr"] = float(state.get("lr", 0.0))
        return rec

    def _append(self, line: str) -> None:
        if self.log_path is None:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
