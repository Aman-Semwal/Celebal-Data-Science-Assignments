"""Project-wide logging configuration.

Usage:
    from src.utils.logging_utils import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def _configure_root(log_file: Path | None = None, level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(
    name: str, log_file: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Return a module-level logger, configuring the root logger on first call.

    Args:
        name: Usually ``__name__`` of the calling module.
        log_file: Optional path to also write logs to a file.
        level: Logging level for the root logger (only applied on first call).

    Returns:
        A configured ``logging.Logger`` instance.
    """
    _configure_root(log_file=log_file, level=level)
    return logging.getLogger(name)
