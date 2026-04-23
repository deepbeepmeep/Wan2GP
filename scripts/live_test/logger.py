"""Small logging shim for the live-test harness."""

from __future__ import annotations

import logging
from typing import Any


_CONFIGURED = False


def configure_logging(*, level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _CONFIGURED = True

    try:
        import structlog
    except ImportError:
        return

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(resolved_level),
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "scripts.live_test") -> Any:
    configure_logging()
    try:
        import structlog
    except ImportError:
        return logging.getLogger(name)
    return structlog.get_logger(name)


__all__ = ["configure_logging", "get_logger"]
