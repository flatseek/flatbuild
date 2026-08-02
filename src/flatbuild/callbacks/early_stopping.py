"""Early stopping callback."""

from __future__ import annotations

from flatbuild.callbacks.base import Callback, CallbackContext


class EarlyStoppingCallback(Callback):
    """Stop training when a monitored metric plateaus.

    Args:
        patience: Consecutive evaluation passes without improvement
            before stopping.
        min_delta: Minimum improvement to count as improvement.
        monitor: Metric name to track (default ``"val_loss"``).
        mode: ``"min"`` (loss) or ``"max"`` (accuracy). Defaults to
            ``"min"`` when ``monitor`` contains ``"loss"``, else ``"max"``.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        monitor: str = "val_loss",
        mode: str | None = None,
    ) -> None:
        """Initialize."""
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode or ("min" if "loss" in monitor else "max")
        self._best = None
        self._counter = 0

    def on_eval_end(self, ctx: CallbackContext) -> None:
        """Compare the latest metric to the best so far; possibly stop."""
        state = ctx.state
        metric = state.get(self.monitor)
        if metric is None:
            return
        if self._best is None:
            self._best = metric
            self._counter = 0
            return

        better = (
            metric < self._best - self.min_delta
            if self.mode == "min"
            else metric > self._best + self.min_delta
        )
        if better:
            self._best = metric
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                ctx.trainer._stop_training = True
                ctx.state["early_stopped"] = True
                self.on_early_stop(ctx)

    def on_early_stop(self, ctx: CallbackContext) -> None:
        """Hook override: log early stopping."""
        from flatbuild.utils import get_logger

        logger = get_logger(__name__)
        logger.info(
            f"Early stopping triggered (patience={self.patience}, "
            f"monitor={self.monitor}, best={self._best})"
        )
