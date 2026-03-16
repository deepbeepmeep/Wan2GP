"""Lightweight debug-diagnostics helpers used by CLI/tests."""

from __future__ import annotations

import re

_EDGE_FAILURE_RE = re.compile(
    r"\[EDGE_FAIL:(?P<function>[^:\]]+):HTTP_(?P<status>\d{3})\]\s*(?P<message>.*)"
)


def parse_edge_failure(text: str):
    """Parse the structured edge-failure marker emitted by runtime code."""
    match = _EDGE_FAILURE_RE.search(text or "")
    if not match:
        return None
    return {
        "function": match.group("function"),
        "status": int(match.group("status")),
        "message": match.group("message"),
    }


def diagnose_error_text(text: str) -> list[str]:
    """Return a short list of human-readable diagnostics for a raw error string."""
    diagnostics: list[str] = []
    parsed = parse_edge_failure(text)
    lowered = (text or "").lower()

    if parsed is not None:
        if 500 <= parsed["status"] < 600:
            diagnostics.append("Detected edge-function 5xx error.")
            diagnostics.append("This often means the upload failed or the edge handler crashed.")
        else:
            diagnostics.append(f"Detected edge-function failure in {parsed['function']}.")

    if "cuda out of memory" in lowered or "out of memory" in lowered:
        diagnostics.append("Detected GPU memory error.")

    if not diagnostics:
        diagnostics.append("No specific diagnosis available.")
    return diagnostics

