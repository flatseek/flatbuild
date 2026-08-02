"""ProgressReporter — tqdm-based per-epoch progress display.

Per requirements:
- one progress bar per epoch
- ETA, elapsed, current step, total step
- loss, learning rate
- samples/sec, tokens/sec
- optional GPU/CPU memory if available
- clean resume semantics (does not lose its place)
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from tqdm.auto import tqdm


# Set this env var to render `1k`, `1m`, etc. on terminals that
# support Unicode block drawing characters.
os.environ.setdefault("TQDM_USE_UNICODE", "1")


@dataclass
class _StepStats:
    """Rolling counters for a single epoch."""

    step_count: int = 0
    sample_count: int = 0
    token_count: int = 0
    last_loss: float | None = None
    last_lr: float | None = None
    last_step_time: float = 0.0


def _format_seconds(seconds: float) -> str:
    """Format ``seconds`` as ``HH:MM:SS`` or ``MM:SS``."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _human_count(value: float) -> str:
    """Format ``value`` as a short, human-readable count.

    Args:
        value: Numeric count.

    Returns:
        ``"123"``, ``"1.2k"``, ``"5m"``, etc. Tiny values
        (``0 < abs(v) < 1``) switch to scientific notation so a
        value like ``3e-5`` shows as ``3.0e-05`` instead of being
        rounded to ``"0"``.
    """
    if value == 0:
        return "0.0"
    sign = ""
    if value < 0:
        sign = "-"
        value = -value
    if 0 < value < 1.0:
        return f"{sign}{value:.1e}"
    units = ("", "k", "m", "b")
    idx = 0
    while idx < len(units) - 1 and value >= 1000:
        value /= 1000
        idx += 1
    unit = units[idx]
    if idx == 0 and value >= 100:
        return f"{sign}{value:.0f}"
    return f"{sign}{value:.1f}{unit}"


class ProgressReporter:
    """Per-epoch tqdm wrapper with rolling throughput metrics.

    Args:
        step: When non-zero, the progress bar starts at this step
            (used after a checkpoint resume so the bar reflects the
            real global position).
    """

    def __init__(self, *, start_step: int = 0) -> None:
        """Initialize state. No bar is shown yet."""
        self.start_step = int(start_step)
        self._stats = _StepStats()
        self._bar: tqdm | None = None
        self._epoch_start: float = 0.0

    def start_epoch(self, total: int, *, epoch: int, total_epochs: int, desc: str | None = None) -> tqdm:
        """Open a fresh progress bar for one epoch.

        Args:
            total: Total steps in this epoch (post-shuffle).
            epoch: 1-based epoch index.
            total_epochs: Number of epochs in the run.
            desc: Optional bar description. Defaults to ``"Epoch X/Y"``.

        Returns:
            The underlying :class:`tqdm.auto.tqdm` handle (mainly for tests).
        """
        self._stats = _StepStats(step_count=self.start_step)
        self._epoch_start = time.perf_counter()
        prefix = desc or f"Epoch {epoch}/{total_epochs}"
        self._bar = tqdm(
            total=max(1, total),
            desc=prefix,
            dynamic_ncols=True,
            smoothing=0.05,
            mininterval=0.25,  # respect `min_interval` for log spam
            maxinterval=5.0,
            unit="step",
        )
        # ``start_step`` semantics — collapse already-completed steps so
        # we can't accidentally overrun the bar on resume.
        if self.start_step > 0:
            self._bar.update(self.start_step)
        return self._bar

    def update(
        self,
        *,
        step: int,
        loss: float | None,
        lr: float | None,
        samples: int,
        tokens: int,
        elapsed: float | None = None,
    ) -> None:
        """Advance the bar by one step and refresh postfix metrics.

        Args:
            step: 1-based step count for this update (used for
                positional tracking only).
            loss: Latest training loss to display.
            lr: Current learning rate.
            samples: Samples processed since the previous update.
            tokens: Tokens processed since the previous update.
            elapsed: Optional explicit elapsed (defaults to perf_counter
                since epoch start).
        """
        if self._bar is None:
            return
        self._stats.step_count = step
        self._stats.sample_count += samples
        self._stats.token_count += tokens
        self._stats.last_loss = loss if loss is not None else self._stats.last_loss
        self._stats.last_lr = lr if lr is not None else self._stats.last_lr
        elapsed_v = elapsed if elapsed is not None else (time.perf_counter() - self._epoch_start)
        avg = elapsed_v / max(1, step - self.start_step)
        # Per-step throughput
        self._stats.last_step_time = avg
        samples_per_s = samples / max(1e-6, avg)
        tokens_per_s = tokens / max(1e-6, avg)
        postfix: dict[str, Any] = {
            "loss": f"{self._stats.last_loss:.4f}" if self._stats.last_loss is not None else "?",
            "lr": _human_count(self._stats.last_lr) if self._stats.last_lr is not None else "?",
            "samples/s": f"{samples_per_s:.0f}" if avg > 0 else "?",
            "tokens/s": _human_count(tokens_per_s),
        }
        mem = self._memory()
        if mem:
            postfix.update(mem)
        postfix.update({
            "elapsed": _format_seconds(elapsed_v),
            "eta": _format_seconds(avg * max(0, (self._bar.total - step))),
        })
        self._bar.set_postfix(postfix, refresh=False)
        self._bar.update(1)

    def close(self) -> None:
        """Close the active bar if any."""
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    # ------- internal helpers -------

    @staticmethod
    def _memory() -> dict[str, str]:
        """Return ``{cpu_mem, gpu_mem}`` strings if both available.

        Always returns at least CPU RAM (via :mod:`resource`). GPU is
        reported when :mod:`torch.cuda` is available.
        """
        out: dict[str, str] = {}
        # CPU
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On macOS the unit is bytes; on Linux it's kilobytes.
            mb = usage / 1024 / 1024 if usage > 10**6 else usage / 1024
            out["cpu_mem"] = f"{mb:.0f}M"
        except (ImportError, OSError, ValueError):
            pass
        # GPU
        try:
            import torch

            if torch.cuda.is_available():
                # Reserved memory is a stable high-water mark for memory use.
                reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
                out["gpu_mem"] = f"{reserved:.0f}M"
        except Exception:  # pragma: no cover - defensive
            pass
        return out


__all__ = ["ProgressReporter"]
