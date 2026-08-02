"""PerformanceProfiler — measures wall-clock time per training phase.

The profiler intentionally uses :func:`time.perf_counter` (monotonic,
nanosecond precision) and a tiny inline ``with`` context manager
instead of a heavyweight library dependency.

Phases tracked out of the box:

- ``data``     — moving a batch to device (and any DataLoader fetch overhead).
- ``forward``  — the model forward pass (incl. loss computation).
- ``backward`` — ``loss.backward()``.
- ``optim``    — the optimizer and (optional) scheduler step.
- ``ckpt``     — checkpoint serialisation + flush.
- ``valid``    — validation pass.

The user can also record arbitrary named phases via
:meth:`PerformanceProfiler.measure`.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from flatbuild.utils import get_logger

logger = get_logger(__name__)


@dataclass
class _PhaseTotals:
    """Accumulator for a single phase."""

    total: float = 0.0
    count: int = 0

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass
class PerformanceProfiler:
    """Aggregating profiler with a context-manager API.

    Args:
        enabled: When ``False``, all timing context managers become
            no-ops (cheap). Tests use this to keep test runtime small.
    """

    enabled: bool = True
    _phases: dict[str, _PhaseTotals] = field(default_factory=lambda: defaultdict(_PhaseTotals))
    _step_count: int = 0

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Time a block under ``name``.

        Args:
            name: Phase label (e.g. ``"forward"``).

        Yields:
            ``None`` — use as ``with profiler.measure("forward"):``.
        """
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            bucket = self._phases[name]
            bucket.total += elapsed
            bucket.count += 1

    def tick(self, *, samples: int = 0, tokens: int = 0) -> None:
        """Increment step counter and accumulate samples / tokens.

        Args:
            samples: Samples processed since last tick.
            tokens: Tokens processed since last tick.
        """
        self._step_count += 1

    def summary(self) -> dict[str, float]:
        """Compute the per-phase percentage breakdown.

        Returns:
            Dictionary mapping phase name → ``{total, count, avg, pct}``.
            ``pct`` is the proportion of measured time (sum of all
            ``total`` fields); ``avg`` is mean per-call seconds.
        """
        grand_total = sum(b.total for b in self._phases.values())
        out: dict[str, float] = {}
        for name, bucket in self._phases.items():
            pct = (bucket.total / grand_total * 100.0) if grand_total > 0 else 0.0
            out[name] = {
                "total": bucket.total,
                "count": bucket.count,
                "avg": bucket.avg,
                "pct": pct,
            }
        return out

    def print_summary(self, *, step_count: int | None = None, elapsed: float | None = None) -> None:
        """Print a formatted breakdown of measured phases.

        Args:
            step_count: Total optimizer steps in the run (used for
                per-step averages). Defaults to the tick counter.
            elapsed: Optional explicit wall-clock for the run.
        """
        if not self.enabled:
            return
        s = self.summary()
        if not s:
            logger.info("PerformanceProfiler: no phases recorded.")
            return

        steps = step_count if step_count is not None else self._step_count
        if elapsed is None:
            elapsed = sum(p["total"] for p in s.values())

        rows = sorted(s.items(), key=lambda kv: -kv[1]["total"])
        logger.info("=" * 56)
        logger.info("Flatbuild performance breakdown")
        logger.info("=" * 56)
        for name, p in rows:
            logger.info(f"{name:<22} {p['pct']:>6.1f}%  total={p['total']:.3f}s  calls={p['count']}")
        logger.info("-" * 56)
        if steps and elapsed:
            avg = elapsed / max(1, steps)
            samples_s = 0.0
            tokens_s = 0.0
            for name in ("data", "forward", "backward", "optim"):
                p = s.get(name)
                if p and p["count"]:
                    continue
            logger.info(f"Steps............... {steps:,}")
            logger.info(f"Wall-clock.......... {elapsed:.1f}s")
            logger.info(f"Avg / step.......... {avg * 1000:.2f}ms")
        logger.info("=" * 56)


__all__ = ["PerformanceProfiler"]
