"""Structured logging for Pcap2Rule experiments."""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "pcap2rule",
    log_dir: Optional[str] = None,
    level: str = "INFO",
    to_console: bool = True,
) -> logging.Logger:
    """Configure and return a logger with file and console handlers.

    Args:
        name: Logger name.
        log_dir: Directory for log files (creates timestamped file if provided).
        level: Logging level string.
        to_console: Whether to also log to stdout.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    if to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{timestamp}.log"),
            encoding='utf-8',
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "pcap2rule") -> logging.Logger:
    """Get or create a logger instance (convenience wrapper)."""
    return logging.getLogger(name)


# Alias for backward compatibility
setup_logging = setup_logger
