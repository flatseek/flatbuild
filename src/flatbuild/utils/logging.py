"""Logging utilities for Flatbuild."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    stage: str | None = None,
    level: int = logging.INFO,
    log_dir: Path | None = None,
) -> None:
    """Configure root logging for a Flatbuild run.

    Args:
        stage: Optional stage name (e.g. ``"train"``). When provided,
            output is also mirrored to ``<log_dir>/<stage>.log``.
        level: Logging level for the root logger.
        log_dir: Directory for log files. Defaults to ``./logs``.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if stage:
        target_dir = Path(log_dir) if log_dir else Path("logs")
        target_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(target_dir / f"{stage}.log"))

    logging.basicConfig(
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        level=level,
        handlers=handlers,
        force=True,
    )

    # Quiet down chatty third-party loggers.
    for name in ("transformers", "datasets", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A standard-library logger instance.
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager that prefixes messages with a stage tag."""

    def __init__(
        self,
        logger: logging.Logger,
        stage: str,
        message: str | None = None,
    ) -> None:
        """Initialize the log context.

        Args:
            logger: Logger to use.
            stage: Stage tag (e.g. ``"train"``).
            message: Optional message to log on entry.
        """
        self.logger = logger
        self.stage = stage
        self.message = message

    def __enter__(self) -> "LogContext":
        """Enter the context."""
        if self.message:
            self.logger.info(f"[{self.stage}] {self.message}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the context. Logs but does not suppress exceptions."""
        if exc_type:
            self.logger.error(f"[{self.stage}] {exc_val}")
        return False
