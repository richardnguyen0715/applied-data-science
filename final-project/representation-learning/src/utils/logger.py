"""Logging utilities for console and file outputs."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(
    name: str,
    log_dir: str = "logs",
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger with synchronized console and file handlers.

    Args:
        name: Logger name.
        log_dir: Output directory for log files.
        log_file: Explicit log filename. If not provided, ``{name}.log`` is used.
        level: Logging level applied to the logger and handlers.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    resolved_log_file = log_file if log_file is not None else f"{name}.log"

    file_handler = logging.FileHandler(log_path / resolved_log_file)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
