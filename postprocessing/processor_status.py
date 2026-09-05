"""Shared availability status contract for post-processing handlers."""

from __future__ import annotations

from typing import Any


PROCESSOR_STATUS_ENABLED = "enabled"
PROCESSOR_STATUS_DISABLED = "disabled"
PROCESSOR_STATUS_UNKNOWN = "unknown"
PROCESSOR_STATUSES = (PROCESSOR_STATUS_ENABLED, PROCESSOR_STATUS_DISABLED, PROCESSOR_STATUS_UNKNOWN)


def handler_status(handler: Any) -> str:
    if hasattr(handler, "enabled"):
        return PROCESSOR_STATUS_ENABLED if handler.enabled() else PROCESSOR_STATUS_DISABLED
    value = str(getattr(handler, "status", PROCESSOR_STATUS_UNKNOWN) or "").strip().lower()
    return value if value in PROCESSOR_STATUSES else PROCESSOR_STATUS_UNKNOWN


def handler_reason_disabled(handler: Any) -> str:
    if handler_status(handler) != PROCESSOR_STATUS_DISABLED:
        return ""
    return str(getattr(handler, "reason_disabled", "") or "").strip()
