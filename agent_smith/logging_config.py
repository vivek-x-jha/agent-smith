"""Centralized structlog configuration for the service."""

from __future__ import annotations

import logging
from logging.config import dictConfig
from typing import Any

import structlog

from .config import get_settings


def setup_logging(level: str | int | None = None) -> None:
    """Configure JSON logging with structlog bound to stdlib logging."""

    if level is None:
        settings = get_settings()
        level = "DEBUG" if settings.env == "development" else "INFO"

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "plain": {
                    "format": "%(message)s",
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "plain",
                    "level": level,
                }
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": level,
                }
            },
        }
    )

    timestamper = structlog.processors.TimeStamper(fmt="iso")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level) if isinstance(level, str) else level
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger (optionally named)."""

    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


__all__ = ["setup_logging", "get_logger"]
