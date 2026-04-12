"""Centralized logging utility module."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


class ColorFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color."""
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = self.COLORS[levelname] + levelname + self.RESET
        return super().format(record)


def get_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).
        log_dir: Directory to save log files. If None, only console logging.
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with color
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = ColorFormatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if log_dir is provided
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f'{name}.log',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
