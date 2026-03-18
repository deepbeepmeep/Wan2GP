"""Common log formatting helpers."""

from __future__ import annotations


def safe_log_fields(log: dict, *, message_limit: int = 120) -> dict[str, str]:
    timestamp = str(log.get("timestamp", ""))
    return {
        "timestamp": timestamp[11:19] if len(timestamp) >= 19 else timestamp,
        "level": str(log.get("log_level", "")),
        "message": (
            str(log.get("message", ""))[:message_limit] + "..."
            if len(str(log.get("message", ""))) > message_limit
            else str(log.get("message", ""))
        ),
    }


__all__ = ["safe_log_fields"]
