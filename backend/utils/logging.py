"""Logging configuration for the application."""

import logging
import sys

# Singleton logger instance
_logger: logging.Logger | None = None


def _create_logger() -> logging.Logger:
    """Create and configure the singleton logger."""
    log = logging.getLogger("dts")

    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)

    return log


def get_logger() -> logging.Logger:
    """Get the singleton logger instance."""
    global _logger
    if _logger is None:
        _logger = _create_logger()
    return _logger


logger = get_logger()
