"""Worker-detail formatter helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from debug.formatters import Formatter


format_worker = Formatter.format_worker


def get_worker_heartbeat(timestamp: str | None):
    if not timestamp or timestamp == "never":
        return "❌", "never"
    try:
        heartbeat = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
        if age < 60:
            return "✅", f"{age:.0f}s"
        if age < 300:
            return "⚠️", f"{age:.0f}s"
    except Exception:
        return "❌", "invalid"
    return "❌", "stale"


__all__ = ["format_worker", "get_worker_heartbeat"]
