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


DIRECT_ROUTE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "z_image": "z_image_turbo",
        "z_image_turbo": "z_image_turbo",
        "z_image_turbo_i2i": "z_image_turbo_i2i",
        "qwen_image": "qwen_image",
        "qwen_image_2512": "qwen_image_2512",
        "optimised_t2i": "wan_2_2_t2i",
        "wan_2_2_t2i": "wan_2_2_t2i",
        "qwen_image_edit": "qwen_image_edit",
        "qwen_image_style": "qwen_image_style",
        "image_inpaint": "image_inpaint",
        "annotated_image_edit": "annotated_image_edit",
    }
)

SPRINT_2_SELECTOR_MAP: Mapping[str, RouteSelectorEntry] = MappingProxyType(
    {
        "z_image_turbo": RouteSelectorEntry(
            route_key="z_image_turbo",
            support_state=RouteSupportState.VIBECOMFY_SUPPORTED,
            template_id="image/z_image",
            default_resolution="1024x1024",
        ),
        "z_image_turbo_i2i": RouteSelectorEntry(
            route_key="z_image_turbo_i2i",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
        ),
        "qwen_image_2512": RouteSelectorEntry(
            route_key="qwen_image_2512",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
        ),
        "qwen_image": RouteSelectorEntry(
            route_key="qwen_image",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
        ),
        "qwen_image_edit": RouteSelectorEntry(
            route_key="qwen_image_edit",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
        ),
        "qwen_image_style": RouteSelectorEntry(
            route_key="qwen_image_style",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
        ),
        "image_inpaint": RouteSelectorEntry(
            route_key="image_inpaint",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
        ),
        "annotated_image_edit": RouteSelectorEntry(
            route_key="annotated_image_edit",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
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
        "join_clips_segment": RouteSelectorEntry(
            route_key="join_clips_segment",
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
    if source_task_type in _DIMENSIONAL_CHILD_TASK_TYPES:
        return _dimensional_child_route_key(str(source_task_type), task_params)

    if task_type in _DIMENSIONAL_CHILD_TASK_TYPES:
        return _dimensional_child_route_key(task_type, task_params)

    return _direct_route_key(task_type)


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


def route_snapshot_fields(
    *,
    task_type: str,
    params: Mapping[str, Any] | None = None,
    task_id: str | None = None,
    backend: WorkerBackend | str | None = None,
    selector_namespace: str = "production",
    selector_version: int | str | None = None,
    parent_route_key: str | None = None,
) -> dict[str, Any]:
    """Return top-level task route fields plus a JSON-safe snapshot.

    The database migration added these columns for create-time observability and
    later child-row pinning. Live claim authorization still belongs to the
    selector/capability RPCs; this helper only serializes the route decision a
    caller already intends to persist.
    """

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

    snapshot: dict[str, Any] = {
        "selector_namespace": selector_namespace,
        "route_key": route_key,
        "selected_backend": selected_backend.value,
        "selector_version": selector_version,
        "support_state": support_state.value,
        "template_id": template_id,
    }
    if task_id is not None:
        snapshot["task_id"] = task_id
    if parent_route_key is not None:
        snapshot["parent_route_key"] = parent_route_key

    return {
        "selector_namespace": selector_namespace,
        "route_key": route_key,
        "selected_backend": selected_backend.value,
        "selector_version": selector_version,
        "route_selection_snapshot": snapshot,
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
            "join_clips_segment__",
        )
    )


_DIMENSIONAL_CHILD_TASK_TYPES = frozenset(
    {
        "travel_segment",
        "individual_travel_segment",
        "join_clips_segment",
    }
)


def _direct_route_key(task_type: str) -> str:
    slugged = _slug(task_type)
    return DIRECT_ROUTE_ALIASES.get(slugged, task_type)


def _dimensional_child_route_key(task_type: str, params: Mapping[str, Any]) -> str:
    """Derive the Cohort E child key without importing comparison scripts."""

    return "__".join(
        [
            _slug(task_type),
            f"model-{_slug(_route_model_family(params))}",
            f"guidance-{_slug(_route_guidance_kind(task_type, params))}",
            f"continuity-{_slug(_route_continuity_case(task_type, params))}",
            f"profile-{_slug(_route_profile(params))}",
        ]
    )


def _slug(value: Any) -> str:
    text = str(value or "none").strip().lower()
    text = text.replace("+", "_plus_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "none"


def _route_model_family(params: Mapping[str, Any]) -> str:
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


def _route_guidance_kind(task_type: str, params: Mapping[str, Any]) -> str:
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
    if task_type == "join_clips_segment" and _route_model_family(params) == "wan22_vace":
        return "vace"

    return "none"


def _route_continuity_case(task_type: str, params: Mapping[str, Any]) -> str:
    explicit_case = params.get("continuity_case")
    if explicit_case:
        return str(explicit_case)
    if task_type == "join_clips_segment":
        return "join_bridge"
    if params.get("video_source"):
        return "video_source"
    return "first_last"


def _route_profile(params: Mapping[str, Any]) -> str:
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
    "DIRECT_ROUTE_ALIASES",
    "ResolvedTask",
    "RouteSelectorEntry",
    "RouteSupportState",
    "SPRINT_2_SELECTOR_MAP",
    "WorkerBackend",
    "derive_route_key",
    "parse_worker_backend",
    "resolve_task_route",
    "route_snapshot_fields",
    "route_support_state",
    "routing_telemetry_fields",
]
