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
        "error_type": f"HTTP_{match.group('status')}",
        "details": match.group("message"),
    }


def diagnose_error_text(text: str) -> list[str]:
    """Return a short list of human-readable diagnostics for a raw error string."""
    diagnostics: list[str] = []
    parsed = parse_edge_failure(text)
    lowered = (text or "").lower()

    if parsed is not None:
        diagnostics.append(f"DIAGNOSIS: edge function {parsed['function']} failed")
        if parsed["error_type"] in {"HTTP_500", "HTTP_502", "HTTP_503", "HTTP_504"}:
            diagnostics.append("Detected 5xx error from edge function.")
            diagnostics.append("This looks transient and often means upload failed or the edge handler crashed.")
        else:
            diagnostics.append(f"Detected edge-function failure in {parsed['function']}.")

    if "cuda out of memory" in lowered or "out of memory" in lowered:
        diagnostics.append("Detected GPU memory error.")

    if not diagnostics:
        diagnostics.append("No specific diagnosis available.")
    return diagnostics


def extract_lora_urls(params: dict) -> list[str]:
    urls: list[str] = []
    phase_config = params.get("phase_config", {}) if isinstance(params, dict) else {}
    if isinstance(phase_config, dict):
        for phase in phase_config.get("phases", []):
            if not isinstance(phase, dict):
                continue
            for lora in phase.get("loras", []):
                if isinstance(lora, dict) and lora.get("url") and lora["url"] not in urls:
                    urls.append(lora["url"])
    additional_loras = params.get("additional_loras", {}) if isinstance(params, dict) else {}
    if isinstance(additional_loras, dict):
        for url in additional_loras:
            if url not in urls:
                urls.append(url)
    return urls
