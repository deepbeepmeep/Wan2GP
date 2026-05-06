"""Worker-local backend route selection for Sprint 2 VibeComfy support.

This module is intentionally lightweight: it must be importable before WGP queue
conversion and without importing VibeComfy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
import re
from types import MappingProxyType
from typing import Any, Mapping


class RouteSupportState(str, Enum):
    WGP_ONLY = "wgp_only"
    VIBECOMFY_SUPPORTED = "vibecomfy_supported"
    VIBECOMFY_UNSUPPORTED = "vibecomfy_unsupported"


class WorkerBackend(str, Enum):
    WGP = "wgp"
    VIBECOMFY = "vibecomfy"


@dataclass(frozen=True)
class RouteSelectorEntry:
    route_key: str
    support_state: RouteSupportState
    template_id: str | None = None
    default_resolution: str | None = None


@dataclass(frozen=True)
class ResolvedTask:
    task_id: str
    task_type: str
    route_key: str
    backend: WorkerBackend
    support_state: RouteSupportState
    params: Mapping[str, Any] = field(default_factory=dict)
    template_id: str | None = None
    memory_profile: str | None = None
    fail_closed_reason: str | None = None

    @property
    def should_use_vibecomfy(self) -> bool:
        return (
            self.backend == WorkerBackend.VIBECOMFY
            and self.support_state == RouteSupportState.VIBECOMFY_SUPPORTED
            and self.fail_closed_reason is None
        )


SPRINT_2_SELECTOR_MAP: Mapping[str, RouteSelectorEntry] = MappingProxyType(
    {
        "z_image_turbo": RouteSelectorEntry(
            route_key="z_image_turbo",
            support_state=RouteSupportState.VIBECOMFY_SUPPORTED,
            template_id="image/z_image",
            default_resolution="1024x1024",
        ),
        "qwen_image_2512": RouteSelectorEntry(
            route_key="qwen_image_2512",
            support_state=RouteSupportState.VIBECOMFY_SUPPORTED,
            template_id="image/qwen_image_2512",
            default_resolution="768x768",
        ),
        "travel_segment": RouteSelectorEntry(
            route_key="travel_segment",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            template_id=None,
        ),
        "individual_travel_segment": RouteSelectorEntry(
            route_key="individual_travel_segment",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            template_id=None,
        ),
        "wan_2_2_t2i": RouteSelectorEntry(
            route_key="wan_2_2_t2i",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
        ),
    }
)


def parse_worker_backend(value: str | None = None) -> WorkerBackend:
    raw_backend = os.environ.get("REIGH_BACKEND") if value is None else value
    normalized = (raw_backend or "").strip().lower()
    if normalized in {"", "wgp"}:
        return WorkerBackend.WGP
    if normalized == "vibecomfy":
        return WorkerBackend.VIBECOMFY
    raise ValueError(
        "Unsupported REIGH_BACKEND value "
        f"{raw_backend!r}; expected unset, 'wgp', or 'vibecomfy'"
    )


def derive_route_key(task_type: str, params: Mapping[str, Any] | None = None) -> str:
    task_params = params or {}

    source_task_type = task_params.get("_source_task_type")
    if source_task_type in {"travel_segment", "individual_travel_segment"}:
        return _travel_route_key(str(source_task_type), task_params)

    if task_type in {"travel_segment", "individual_travel_segment"}:
        return _travel_route_key(task_type, task_params)

    return task_type


def resolve_task_route(
    *,
    task_id: str,
    task_type: str,
    params: Mapping[str, Any] | None = None,
    backend: WorkerBackend | str | None = None,
) -> ResolvedTask:
    task_params = dict(params or {})
    selected_backend = _coerce_backend(backend) if backend is not None else parse_worker_backend()
    route_key = derive_route_key(task_type, task_params)
    selector_entry = _selector_entry_for_route_key(route_key)
    support_state = (
        selector_entry.support_state
        if selector_entry is not None
        else RouteSupportState.VIBECOMFY_UNSUPPORTED
    )
    template_id = selector_entry.template_id if selector_entry is not None else None
    fail_closed_reason = _fail_closed_reason(
        backend=selected_backend,
        route_key=route_key,
        selector_entry=selector_entry,
        params=task_params,
    )

    return ResolvedTask(
        task_id=task_id,
        task_type=task_type,
        route_key=route_key,
        backend=selected_backend,
        support_state=support_state,
        params=MappingProxyType(task_params),
        template_id=template_id,
        memory_profile=_extract_memory_profile(task_params),
        fail_closed_reason=fail_closed_reason,
    )


def route_support_state(route_key: str) -> RouteSupportState:
    selector_entry = _selector_entry_for_route_key(route_key)
    if selector_entry is None:
        return RouteSupportState.VIBECOMFY_UNSUPPORTED
    return selector_entry.support_state


def routing_telemetry_fields(resolved: ResolvedTask) -> dict[str, str | None]:
    """Return compact structured fields for backend route telemetry."""

    return {
        "task_id": resolved.task_id,
        "task_type": resolved.task_type,
        "route_key": resolved.route_key,
        "backend": resolved.backend.value,
        "template_id": resolved.template_id,
        "support_state": resolved.support_state.value,
        "memory_profile": resolved.memory_profile,
    }


def _coerce_backend(backend: WorkerBackend | str) -> WorkerBackend:
    if isinstance(backend, WorkerBackend):
        return backend
    return parse_worker_backend(backend)


def _selector_entry_for_route_key(route_key: str) -> RouteSelectorEntry | None:
    selector_entry = SPRINT_2_SELECTOR_MAP.get(route_key)
    if selector_entry is not None:
        return selector_entry

    if _is_dimensional_travel_route_key(route_key):
        return RouteSelectorEntry(
            route_key=route_key,
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            template_id=None,
        )

    return None


def _is_dimensional_travel_route_key(route_key: str) -> bool:
    return route_key.startswith(
        (
            "travel_segment__",
            "individual_travel_segment__",
        )
    )


def _travel_route_key(task_type: str, params: Mapping[str, Any]) -> str:
    """Derive the Cohort E child key without importing comparison scripts."""

    return "__".join(
        [
            _slug(task_type),
            f"model-{_slug(_travel_model_family(params))}",
            f"guidance-{_slug(_travel_guidance_kind(params))}",
            f"continuity-{_slug(_travel_continuity_case(params))}",
            f"profile-{_slug(_travel_profile(params))}",
        ]
    )


def _slug(value: Any) -> str:
    text = str(value or "none").strip().lower()
    text = text.replace("+", "_plus_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "none"


def _travel_model_family(params: Mapping[str, Any]) -> str:
    explicit_family = params.get("model_family")
    if explicit_family:
        return str(explicit_family)

    normalized = _slug(params.get("model_name") or params.get("model"))
    if not normalized or normalized == "none":
        return "unknown"
    if "wan_2_2" in normalized or "wan22" in normalized:
        return "wan22_vace" if "vace" in normalized else "wan22_i2v"
    if "ltx2" in normalized:
        return "ltx2_distilled" if "distilled" in normalized else "ltx2"
    if "qwen" in normalized:
        return "qwen"
    if "z_image" in normalized:
        return "z_image"
    return normalized


def _travel_guidance_kind(params: Mapping[str, Any]) -> str:
    explicit_kind = params.get("guidance_kind") or params.get("travel_guidance_kind")
    if explicit_kind:
        return str(explicit_kind)

    travel_guidance = params.get("travel_guidance")
    if isinstance(travel_guidance, Mapping):
        kind = travel_guidance.get("kind")
        if kind:
            return str(kind)

    if params.get("use_uni3c") or params.get("uni3c_guide_video"):
        return "uni3c"
    if params.get("svi2pro"):
        return "vace"
    if params.get("video_guide") or params.get("video_mask"):
        return "vace"

    return "none"


def _travel_continuity_case(params: Mapping[str, Any]) -> str:
    explicit_case = params.get("continuity_case")
    if explicit_case:
        return str(explicit_case)
    if params.get("video_source"):
        return "video_source"
    return "first_last"


def _travel_profile(params: Mapping[str, Any]) -> str:
    return str(
        params.get("profile")
        or params.get("wgp_profile")
        or params.get("override_profile")
        or "default"
    )


def _fail_closed_reason(
    *,
    backend: WorkerBackend,
    route_key: str,
    selector_entry: RouteSelectorEntry | None,
    params: Mapping[str, Any],
) -> str | None:
    if backend == WorkerBackend.WGP:
        return None

    if selector_entry is None:
        return f"Route {route_key!r} has no VibeComfy selector entry"

    if selector_entry.support_state != RouteSupportState.VIBECOMFY_SUPPORTED:
        return (
            f"Route {route_key!r} is {selector_entry.support_state.value}; "
            "explicit VibeComfy backend will not fall back to WGP"
        )

    resolution = params.get("resolution")
    if resolution and selector_entry.default_resolution:
        normalized_resolution = _normalize_resolution(resolution)
        if normalized_resolution != selector_entry.default_resolution:
            return (
                f"Route {route_key!r} requested resolution {resolution!r}, "
                f"but Sprint 2 VibeComfy support is limited to "
                f"{selector_entry.default_resolution}"
            )

    return None


def _normalize_resolution(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return f"{int(value[0])}x{int(value[1])}"

    text = str(value).strip().lower().replace(" ", "")
    if "×" in text:
        text = text.replace("×", "x")
    return text


def _extract_memory_profile(params: Mapping[str, Any]) -> str | None:
    raw_profile = params.get("override_profile")
    if raw_profile is None:
        return None
    return str(raw_profile)


__all__ = [
    "ResolvedTask",
    "RouteSelectorEntry",
    "RouteSupportState",
    "SPRINT_2_SELECTOR_MAP",
    "WorkerBackend",
    "derive_route_key",
    "parse_worker_backend",
    "resolve_task_route",
    "route_support_state",
    "routing_telemetry_fields",
]
