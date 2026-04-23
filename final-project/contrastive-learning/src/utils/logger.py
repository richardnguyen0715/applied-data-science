"""Logging utilities."""

import logging
from pathlib import Path


def setup_logger(
    name: str,
    log_dir: Path = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup logger with console and file handlers.

    Args:
        name: Logger name.
        log_dir: Directory to save log files.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
