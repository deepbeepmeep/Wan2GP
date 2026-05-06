"""Lightweight debug-diagnostics helpers used by CLI/tests."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

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


ROUTE_DEBUG_FIELDS = (
    "route_key",
    "selected_backend",
    "selector_version",
    "support_state",
    "selected_profile",
    "parent_route_key",
)


def extract_task_params(task: dict[str, Any] | None) -> dict[str, Any]:
    """Return task params from dict or JSON-string task rows."""
    if not isinstance(task, dict):
        return {}
    params = task.get("params")
    if params is None:
        params = task.get("task_params")
    if isinstance(params, str):
        try:
            decoded = json.loads(params)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return params if isinstance(params, dict) else {}


def extract_route_contract_summary(task: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the debug-facing route contract fields from a task row."""
    params = extract_task_params(task)
    contract = params.get("route_contract") if isinstance(params.get("route_contract"), dict) else {}
    snapshot = (
        contract.get("route_selection_snapshot")
        if isinstance(contract.get("route_selection_snapshot"), dict)
        else {}
    )
    return {
        "route_key": _coalesce(contract.get("route_key"), snapshot.get("route_key")),
        "selected_backend": _coalesce(
            contract.get("selected_backend"), snapshot.get("selected_backend")
        ),
        "selector_version": _coalesce(
            contract.get("selector_version"), snapshot.get("selector_version")
        ),
        "support_state": _coalesce(snapshot.get("support_state"), contract.get("support_state")),
        "selected_profile": _coalesce(
            contract.get("selected_profile"), snapshot.get("selected_profile")
        ),
        "parent_route_key": _coalesce(snapshot.get("parent_route_key"), contract.get("parent_route_key")),
    }


def format_route_contract_summary(task: dict[str, Any] | None) -> str:
    """Format route contract metadata for compact debug output."""
    summary = extract_route_contract_summary(task)
    if not any(summary.values()):
        return ""
    selector = str(summary["selector_version"] or "")
    return (
        f"route={summary['route_key'] or '-'} "
        f"backend={summary['selected_backend'] or '-'} "
        f"selector_version={selector or '-'} "
        f"support={summary['support_state'] or '-'} "
        f"profile={summary['selected_profile'] or '-'} "
        f"parent={summary['parent_route_key'] or '-'}"
    )


def route_repair_signals(
    task: dict[str, Any] | None,
    *,
    parent_task: dict[str, Any] | None = None,
    siblings: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return read-only repair/debug signals for route-aware lifecycle surfaces."""
    signals: list[str] = []
    if not isinstance(task, dict):
        return signals

    status = str(task.get("status") or "").lower()
    output_location = str(task.get("output_location") or "")
    error_text = f"{task.get('error_message') or ''} {output_location}"
    if "ROUTE_REPAIR_REQUIRED" in error_text:
        signals.append("parent_repair_required")

    if status in {"queued", "pending", "in progress", "in_progress", "processing"}:
        signals.append("partial_child")

    if output_location and status not in {"complete", "completed"} and not _looks_like_error(output_location):
        signals.append("uploaded_but_not_completed")

    parent_summary = extract_route_contract_summary(parent_task)
    task_summary = extract_route_contract_summary(task)
    has_parent_contract = any(parent_summary.values())
    has_task_contract = any(task_summary.values())
    if has_parent_contract and not has_task_contract:
        signals.append("missing_child_route_contract")
    elif has_parent_contract and has_task_contract:
        comparisons = (
            ("selected_backend", "mixed_backend"),
            ("selector_version", "mixed_selector_version"),
            ("selected_profile", "mixed_selected_profile"),
        )
        for field_name, signal in comparisons:
            if task_summary.get(field_name) != parent_summary.get(field_name):
                signals.append(signal)
        expected_parent_route_key = parent_summary.get("route_key")
        if (
            expected_parent_route_key
            and task_summary.get("parent_route_key")
            and task_summary.get("parent_route_key") != expected_parent_route_key
        ):
            signals.append("wrong_parent_route_key")

    if _is_duplicate_completion_candidate(task, siblings or []):
        signals.append("duplicate_completion_candidate")

    return _dedupe_preserving_order(signals)


def _is_duplicate_completion_candidate(
    task: dict[str, Any],
    siblings: list[dict[str, Any]],
) -> bool:
    if not siblings or str(task.get("status") or "").lower() not in {"complete", "completed"}:
        return False
    key = _duplicate_key(task)
    if key is None:
        return False
    complete_counts = Counter(
        _duplicate_key(candidate)
        for candidate in siblings
        if str(candidate.get("status") or "").lower() in {"complete", "completed"}
        and candidate.get("output_location")
        and _duplicate_key(candidate) is not None
    )
    return complete_counts[key] > 1


def _duplicate_key(task: dict[str, Any]) -> tuple[Any, ...] | None:
    params = extract_task_params(task)
    route = extract_route_contract_summary(task)
    position = _coalesce(
        params.get("segment_index"),
        params.get("join_index"),
        params.get("child_order"),
    )
    if position == "":
        return None
    return (task.get("task_type"), position, route.get("route_key") or "")


def _looks_like_error(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("error", "failed", "traceback", "exception"))


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
