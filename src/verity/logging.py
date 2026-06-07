"""Structured logging for verity — day-one infrastructure (ROADMAP Phase 0).

Configure once at process startup with :func:`configure_logging`, then obtain bound
loggers with :func:`get_logger`. Built on structlog so every service emits structured,
greppable events rather than ad-hoc ``print`` calls.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog

__all__ = ["configure_logging", "get_logger"]


def configure_logging(*, level: int = logging.INFO, json_output: bool = False) -> None:
    """Configure process-wide structured logging.

    Call once at startup. ``json_output=True`` emits one JSON object per event (for
    services / production); the default is a human-friendly console renderer.
    """
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        renderer,
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.types.FilteringBoundLogger:
    """Return a bound logger. Call :func:`configure_logging` once before using it."""
    return cast("structlog.types.FilteringBoundLogger", structlog.get_logger(name))
