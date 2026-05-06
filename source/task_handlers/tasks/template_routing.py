"""Worker-local backend route selection for Sprint 2 VibeComfy support.

This module is intentionally lightweight: it must be importable before WGP queue
conversion and without importing VibeComfy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
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


WORKER_ROUTE_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class RouteSelectorEntry:
    route_key: str
    support_state: RouteSupportState
    template_id: str | None = None
    default_resolution: str | None = None
    disposition: str | None = None
    blocking_reason: str | None = None


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


@dataclass(frozen=True)
class ParentChildRoutePreflight:
    ok: bool
    parent_route_key: str | None = None
    child_route_key: str | None = None
    backend: WorkerBackend | None = None
    support_state: RouteSupportState | None = None
    template_id: str | None = None
    selector_namespace: str | None = None
    selector_version: int | str | None = None
    selected_profile: str | None = None
    route_run_id: str | None = None
    worker_contract_version: int | None = None
    fail_closed_reason: str | None = None


@dataclass(frozen=True)
class ChildRouteContractConsistency:
    ok: bool
    fail_closed_reason: str | None = None
    mismatched_task_ids: tuple[str, ...] = ()


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
        "travel_orchestrator": RouteSelectorEntry(
            route_key="travel_orchestrator",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
        ),
        "join_clips_orchestrator": RouteSelectorEntry(
            route_key="join_clips_orchestrator",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
        ),
        "edit_video_orchestrator": RouteSelectorEntry(
            route_key="edit_video_orchestrator",
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
        "travel_stitch": RouteSelectorEntry(
            route_key="travel_stitch",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
        ),
        "join_final_stitch": RouteSelectorEntry(
            route_key="join_final_stitch",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
        ),
        "wan_2_2_t2i": RouteSelectorEntry(
            route_key="wan_2_2_t2i",
            support_state=RouteSupportState.WGP_ONLY,
            template_id=None,
        ),
    }
)


SECTION3A_ROUTE_SUPPORT_MAP: Mapping[str, RouteSelectorEntry] = MappingProxyType(
    {
        "travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            disposition="NEW",
            blocking_reason="Requires the NEW Wan 2.2 VACE cocktail template before Wan-family travel rows can be promoted.",
        ),
        "travel_segment__model-wan22_vace__guidance-vace_flow__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-wan22_vace__guidance-vace_flow__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            disposition="NEW",
            blocking_reason="Requires the NEW Wan 2.2 VACE cocktail template and optical-flow guide preprocessing before promotion.",
        ),
        "travel_segment__model-wan22_vace__guidance-vace_canny__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-wan22_vace__guidance-vace_canny__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            disposition="NEW",
            blocking_reason="Requires the NEW Wan 2.2 VACE cocktail template and Canny guide preprocessing before promotion.",
        ),
        "travel_segment__model-wan22_vace__guidance-vace_depth__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-wan22_vace__guidance-vace_depth__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            disposition="NEW",
            blocking_reason="Requires the NEW Wan 2.2 VACE cocktail template and depth guide handling before promotion.",
        ),
        "travel_segment__model-wan22_vace__guidance-vace_raw__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-wan22_vace__guidance-vace_raw__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            disposition="NEW",
            blocking_reason="Requires the NEW Wan 2.2 VACE cocktail template and raw guide-video passthrough before promotion.",
        ),
        "travel_segment__model-wan22_vace__guidance-uni3c__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-wan22_vace__guidance-uni3c__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            disposition="NEW",
            blocking_reason="Requires the NEW Wan 2.2 VACE cocktail template and Uni3C patch before promotion.",
        ),
        "travel_segment__model-ltx2__guidance-none__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-ltx2__guidance-none__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_SUPPORTED,
            template_id="video/ltx2_3_runexx_first_last_frame",
            disposition="ADAPT",
        ),
        "travel_segment__model-ltx2_distilled__guidance-none__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-ltx2_distilled__guidance-none__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_SUPPORTED,
            template_id="video/ltx2_3_runexx_first_last_frame",
            disposition="ADAPT",
        ),
        "travel_segment__model-ltx2_distilled__guidance-ltx_control_video__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-ltx2_distilled__guidance-ltx_control_video__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            disposition="BLOCKED",
            blocking_reason="The pinned LTX first/last template is not yet proven control-capable for a full-length control guide.",
        ),
        "travel_segment__model-ltx2_distilled__guidance-ltx_control_pose__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-ltx2_distilled__guidance-ltx_control_pose__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            disposition="BLOCKED",
            blocking_reason="The pinned LTX first/last template is not yet proven control-capable for pose-preprocessed full-length guides.",
        ),
        "travel_segment__model-ltx2_distilled__guidance-ltx_control_depth__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-ltx2_distilled__guidance-ltx_control_depth__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            disposition="BLOCKED",
            blocking_reason="The pinned LTX first/last template is not yet proven control-capable for depth-preprocessed full-length guides.",
        ),
        "travel_segment__model-ltx2_distilled__guidance-ltx_control_canny__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-ltx2_distilled__guidance-ltx_control_canny__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            disposition="BLOCKED",
            blocking_reason="The pinned LTX first/last template is not yet proven control-capable for Canny-preprocessed full-length guides.",
        ),
        "travel_segment__model-ltx2_distilled__guidance-ltx_control_cameraman__continuity-first_last__profile-default": RouteSelectorEntry(
            route_key="travel_segment__model-ltx2_distilled__guidance-ltx_control_cameraman__continuity-first_last__profile-default",
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            disposition="BLOCKED",
            blocking_reason="The pinned LTX first/last template is not yet proven control-capable for cameraman full-length guides.",
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


def route_support_report_fields(route_key: str) -> dict[str, str | None]:
    selector_entry = _selector_entry_for_route_key(route_key)
    support_state = (
        selector_entry.support_state
        if selector_entry is not None
        else RouteSupportState.VIBECOMFY_UNSUPPORTED
    )
    return {
        "route_key": route_key,
        "support_state": support_state.value,
        "template_id": selector_entry.template_id if selector_entry is not None else None,
        "disposition": selector_entry.disposition if selector_entry is not None else None,
        "blocking_reason": selector_entry.blocking_reason if selector_entry is not None else None,
    }


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
    profile: str | None = None,
    run_id: str | None = None,
    worker_contract_version: int = WORKER_ROUTE_CONTRACT_VERSION,
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
    selected_profile = str(profile or _route_profile(task_params))

    snapshot: dict[str, Any] = {
        "selector_namespace": selector_namespace,
        "route_key": route_key,
        "selected_backend": selected_backend.value,
        "selector_version": selector_version,
        "support_state": support_state.value,
        "template_id": template_id,
        "selected_profile": selected_profile,
        "route_run_id": run_id,
        "worker_contract_version": worker_contract_version,
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
        "support_state": support_state.value,
        "selected_profile": selected_profile,
        "selected_template_id": template_id,
        "route_run_id": run_id,
        "worker_contract_version": worker_contract_version,
        "route_selection_snapshot": snapshot,
    }


def normalize_route_snapshot_fields(
    route_contract: Mapping[str, Any] | None,
    *,
    task_type: str = "legacy_task",
    params: Mapping[str, Any] | None = None,
    backend: WorkerBackend | str | None = None,
) -> dict[str, Any]:
    """Inflate legacy or partial route contracts to the full snapshot shape."""

    candidate = route_contract if isinstance(route_contract, Mapping) else {}
    snapshot = candidate.get("route_selection_snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    base = route_snapshot_fields(
        task_type=task_type,
        params=params,
        backend=backend or candidate.get("selected_backend") or snapshot.get("selected_backend") or "wgp",
    )

    route_key = _non_empty_str(candidate.get("route_key")) or _non_empty_str(snapshot.get("route_key")) or base["route_key"]
    selected_backend = _coerce_backend(
        candidate.get("selected_backend") or snapshot.get("selected_backend") or base["selected_backend"]
    )
    selector_entry = _selector_entry_for_route_key(route_key)
    fallback_support_state = (
        selector_entry.support_state
        if selector_entry is not None
        else RouteSupportState.VIBECOMFY_UNSUPPORTED
    )
    support_state = _parse_support_state(
        candidate.get("support_state") or snapshot.get("support_state")
    ) or fallback_support_state
    template_id = _nullable_str(candidate.get("selected_template_id"))
    if template_id is None:
        template_id = _nullable_str(snapshot.get("template_id"))
    if template_id is None and selector_entry is not None:
        template_id = selector_entry.template_id

    selector_namespace = (
        _non_empty_str(candidate.get("selector_namespace"))
        or _non_empty_str(snapshot.get("selector_namespace"))
        or base["selector_namespace"]
    )
    selector_version = _selector_version(
        candidate.get("selector_version")
        if candidate.get("selector_version") is not None
        else snapshot.get("selector_version")
    )
    if selector_version is None:
        selector_version = base["selector_version"]
    selected_profile = (
        _non_empty_str(candidate.get("selected_profile"))
        or _non_empty_str(snapshot.get("selected_profile"))
        or base["selected_profile"]
    )
    route_run_id = _nullable_str(candidate.get("route_run_id"))
    if route_run_id is None:
        route_run_id = _nullable_str(snapshot.get("route_run_id"))
    worker_contract_version = (
        _worker_contract_version(candidate.get("worker_contract_version"))
        or _worker_contract_version(snapshot.get("worker_contract_version"))
        or base["worker_contract_version"]
    )

    normalized_snapshot: dict[str, Any] = {
        "selector_namespace": selector_namespace,
        "route_key": route_key,
        "selected_backend": selected_backend.value,
        "selector_version": selector_version,
        "support_state": support_state.value,
        "template_id": template_id,
        "selected_profile": selected_profile,
        "route_run_id": route_run_id,
        "worker_contract_version": worker_contract_version,
    }
    task_id = _non_empty_str(snapshot.get("task_id"))
    parent_route_key = _non_empty_str(snapshot.get("parent_route_key"))
    if task_id is not None:
        normalized_snapshot["task_id"] = task_id
    if parent_route_key is not None:
        normalized_snapshot["parent_route_key"] = parent_route_key

    return {
        "selector_namespace": selector_namespace,
        "route_key": route_key,
        "selected_backend": selected_backend.value,
        "selector_version": selector_version,
        "support_state": support_state.value,
        "selected_profile": selected_profile,
        "selected_template_id": template_id,
        "route_run_id": route_run_id,
        "worker_contract_version": worker_contract_version,
        "route_selection_snapshot": normalized_snapshot,
    }


def preflight_parent_child_route(
    *,
    parent_params: Mapping[str, Any] | None,
    child_task_type: str,
    child_params: Mapping[str, Any] | None = None,
) -> ParentChildRoutePreflight:
    """Validate a DB-created child can inherit its claimed parent's route."""

    parent_contract = _extract_parent_route_contract(parent_params)
    if parent_contract is None:
        return ParentChildRoutePreflight(
            ok=False,
            fail_closed_reason="Missing or malformed parent params.route_contract",
        )

    contract_fields = _parse_parent_route_contract(parent_contract)
    if isinstance(contract_fields, str):
        return ParentChildRoutePreflight(ok=False, fail_closed_reason=contract_fields)

    task_params = dict(child_params or {})
    child_route_key = derive_route_key(child_task_type, task_params)
    selector_entry = _selector_entry_for_route_key(child_route_key)
    support_state = (
        selector_entry.support_state
        if selector_entry is not None
        else RouteSupportState.VIBECOMFY_UNSUPPORTED
    )
    template_id = selector_entry.template_id if selector_entry is not None else None

    fail_closed_reason = _fail_closed_reason(
        backend=contract_fields["backend"],
        route_key=child_route_key,
        selector_entry=selector_entry,
        params=task_params,
    )
    if fail_closed_reason is None and (
        contract_fields["backend"] == WorkerBackend.VIBECOMFY
        and support_state == RouteSupportState.VIBECOMFY_SUPPORTED
        and template_id is None
    ):
        fail_closed_reason = (
            f"Route {child_route_key!r} is missing a VibeComfy template; "
            "explicit VibeComfy backend will not fall back to WGP"
        )

    return ParentChildRoutePreflight(
        ok=fail_closed_reason is None,
        parent_route_key=contract_fields["parent_route_key"],
        child_route_key=child_route_key,
        backend=contract_fields["backend"],
        support_state=support_state,
        template_id=template_id,
        selector_namespace=contract_fields["selector_namespace"],
        selector_version=contract_fields["selector_version"],
        selected_profile=contract_fields["selected_profile"],
        route_run_id=contract_fields["route_run_id"],
        worker_contract_version=contract_fields["worker_contract_version"],
        fail_closed_reason=fail_closed_reason,
    )


def parent_derived_child_route_snapshot_fields(
    *,
    parent_params: Mapping[str, Any] | None,
    child_task_type: str,
    child_params: Mapping[str, Any] | None = None,
    child_task_id: str | None = None,
) -> dict[str, Any]:
    """Build child route snapshot fields from the parent's persisted contract."""

    preflight = preflight_parent_child_route(
        parent_params=parent_params,
        child_task_type=child_task_type,
        child_params=child_params,
    )
    if not preflight.ok:
        raise ValueError(preflight.fail_closed_reason or "Parent-derived child route failed closed")

    assert preflight.backend is not None
    assert preflight.selector_namespace is not None
    assert preflight.selected_profile is not None
    assert preflight.worker_contract_version is not None

    return route_snapshot_fields(
        task_id=child_task_id,
        task_type=child_task_type,
        params=child_params,
        backend=preflight.backend,
        selector_namespace=preflight.selector_namespace,
        selector_version=preflight.selector_version,
        parent_route_key=preflight.parent_route_key,
        profile=preflight.selected_profile,
        run_id=preflight.route_run_id,
        worker_contract_version=preflight.worker_contract_version,
    )


def validate_existing_child_route_contracts(
    *,
    parent_params: Mapping[str, Any] | None,
    child_tasks: list[Mapping[str, Any]],
    expected_parent_route_key: str | None = None,
) -> ChildRouteContractConsistency:
    """Validate existing DB children still match their claimed parent route."""

    parent_contract = _extract_parent_route_contract(parent_params)
    if parent_contract is None:
        return ChildRouteContractConsistency(
            ok=False,
            fail_closed_reason="Missing or malformed parent params.route_contract",
        )

    parsed_parent = _parse_parent_route_contract(parent_contract)
    if isinstance(parsed_parent, str):
        return ChildRouteContractConsistency(ok=False, fail_closed_reason=parsed_parent)

    parent_route_key = expected_parent_route_key or parsed_parent["parent_route_key"]
    mismatches: list[str] = []
    reasons: list[str] = []

    for child in child_tasks:
        child_id = _task_identity_for_contract_error(child)
        child_params = _extract_child_params_for_contract(child)
        child_contract = _extract_parent_route_contract(child_params)
        if child_contract is None:
            mismatches.append(child_id)
            reasons.append(f"{child_id}: missing params.route_contract")
            continue

        parsed_child = _parse_parent_route_contract(child_contract)
        if isinstance(parsed_child, str):
            mismatches.append(child_id)
            reasons.append(f"{child_id}: {parsed_child}")
            continue

        snapshot = child_contract.get("route_selection_snapshot")
        child_parent_route_key = (
            snapshot.get("parent_route_key") if isinstance(snapshot, Mapping) else None
        )
        comparisons = (
            ("selected_backend", parsed_child["backend"], parsed_parent["backend"]),
            ("selector_namespace", parsed_child["selector_namespace"], parsed_parent["selector_namespace"]),
            ("selector_version", parsed_child["selector_version"], parsed_parent["selector_version"]),
            ("selected_profile", parsed_child["selected_profile"], parsed_parent["selected_profile"]),
            ("parent_route_key", child_parent_route_key, parent_route_key),
        )
        mismatched_fields = [
            field_name
            for field_name, actual, expected in comparisons
            if actual != expected
        ]
        if mismatched_fields:
            mismatches.append(child_id)
            reasons.append(f"{child_id}: route contract mismatch in {', '.join(mismatched_fields)}")

    if mismatches:
        return ChildRouteContractConsistency(
            ok=False,
            fail_closed_reason="Existing child route contracts require repair: " + "; ".join(reasons),
            mismatched_task_ids=tuple(mismatches),
        )

    return ChildRouteContractConsistency(ok=True)


def _coerce_backend(backend: WorkerBackend | str) -> WorkerBackend:
    if isinstance(backend, WorkerBackend):
        return backend
    return parse_worker_backend(backend)


def _extract_child_params_for_contract(task: Mapping[str, Any]) -> Mapping[str, Any]:
    params = task.get("params")
    if params is None:
        params = task.get("task_params")
    if isinstance(params, str):
        try:
            decoded = json.loads(params)
        except (json.JSONDecodeError, ValueError, TypeError):
            return {}
        return decoded if isinstance(decoded, Mapping) else {}
    return params if isinstance(params, Mapping) else {}


def _task_identity_for_contract_error(task: Mapping[str, Any]) -> str:
    for key in ("id", "task_id"):
        value = task.get(key)
        if isinstance(value, str) and value:
            return value
    return "<unknown-child>"


def _extract_parent_route_contract(parent_params: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(parent_params, Mapping):
        return None

    route_contract = parent_params.get("route_contract")
    if not isinstance(route_contract, Mapping):
        return None

    return route_contract


def _parse_parent_route_contract(
    route_contract: Mapping[str, Any],
) -> dict[str, Any] | str:
    backend_value = route_contract.get("selected_backend")
    if not isinstance(backend_value, str):
        return "Malformed parent params.route_contract: selected_backend is required"
    try:
        backend = _coerce_backend(backend_value)
    except ValueError:
        return "Malformed parent params.route_contract: selected_backend is invalid"

    parent_route_key = route_contract.get("route_key")
    if not isinstance(parent_route_key, str) or not parent_route_key.strip():
        return "Malformed parent params.route_contract: route_key is required"

    selector_namespace = route_contract.get("selector_namespace")
    if not isinstance(selector_namespace, str) or not selector_namespace.strip():
        return "Malformed parent params.route_contract: selector_namespace is required"

    selected_profile = route_contract.get("selected_profile")
    if not isinstance(selected_profile, str) or not selected_profile.strip():
        snapshot = route_contract.get("route_selection_snapshot")
        selected_profile = (
            snapshot.get("selected_profile")
            if isinstance(snapshot, Mapping)
            else None
        )
    if not isinstance(selected_profile, str) or not selected_profile.strip():
        selected_profile = "default"

    worker_contract_version = route_contract.get("worker_contract_version")
    if not isinstance(worker_contract_version, int):
        snapshot = route_contract.get("route_selection_snapshot")
        worker_contract_version = (
            snapshot.get("worker_contract_version")
            if isinstance(snapshot, Mapping)
            else None
        )
    if not isinstance(worker_contract_version, int):
        worker_contract_version = WORKER_ROUTE_CONTRACT_VERSION

    selector_version = route_contract.get("selector_version")
    if selector_version is not None and not isinstance(selector_version, (int, str)):
        return "Malformed parent params.route_contract: selector_version must be int, string, or null"

    route_run_id = route_contract.get("route_run_id")
    if route_run_id is not None and not isinstance(route_run_id, str):
        return "Malformed parent params.route_contract: route_run_id must be string or null"

    return {
        "backend": backend,
        "parent_route_key": parent_route_key,
        "selector_namespace": selector_namespace,
        "selector_version": selector_version,
        "selected_profile": selected_profile,
        "route_run_id": route_run_id,
        "worker_contract_version": worker_contract_version,
    }


def _selector_entry_for_route_key(route_key: str) -> RouteSelectorEntry | None:
    selector_entry = SPRINT_2_SELECTOR_MAP.get(route_key)
    if selector_entry is not None:
        return selector_entry

    selector_entry = SECTION3A_ROUTE_SUPPORT_MAP.get(route_key)
    if selector_entry is not None:
        return selector_entry

    if _is_dimensional_travel_route_key(route_key):
        return RouteSelectorEntry(
            route_key=route_key,
            support_state=RouteSupportState.VIBECOMFY_UNSUPPORTED,
            template_id=None,
        )

    return None


def _non_empty_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _nullable_str(value: Any) -> str | None:
    return value if value is None or isinstance(value, str) else None


def _selector_version(value: Any) -> int | str | None:
    return value if isinstance(value, (int, str)) else None


def _worker_contract_version(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _parse_support_state(value: Any) -> RouteSupportState | None:
    try:
        return RouteSupportState(value)
    except ValueError:
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
            f"guidance-{_slug(_route_guidance_key(task_type, params))}",
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


def _route_guidance_mode(params: Mapping[str, Any]) -> str:
    explicit_mode = params.get("guidance_mode") or params.get("travel_guidance_mode")
    if explicit_mode:
        return str(explicit_mode)

    travel_guidance = params.get("travel_guidance")
    if isinstance(travel_guidance, Mapping):
        mode = travel_guidance.get("mode")
        if mode:
            return str(mode)

    return "none"


def _route_guidance_key(task_type: str, params: Mapping[str, Any]) -> str:
    kind = _route_guidance_kind(task_type, params)
    mode = _route_guidance_mode(params)
    if kind in {"vace", "ltx_control"} and mode and mode != "none":
        return f"{kind}_{mode}"
    return kind


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

    if selector_entry.template_id is None:
        return (
            f"Route {route_key!r} is VibeComfy-supported but has no template; "
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
    "ChildRouteContractConsistency",
    "ParentChildRoutePreflight",
    "ResolvedTask",
    "RouteSelectorEntry",
    "RouteSupportState",
    "SECTION3A_ROUTE_SUPPORT_MAP",
    "SPRINT_2_SELECTOR_MAP",
    "WorkerBackend",
    "WORKER_ROUTE_CONTRACT_VERSION",
    "derive_route_key",
    "parent_derived_child_route_snapshot_fields",
    "normalize_route_snapshot_fields",
    "parse_worker_backend",
    "preflight_parent_child_route",
    "resolve_task_route",
    "route_snapshot_fields",
    "route_support_report_fields",
    "route_support_state",
    "routing_telemetry_fields",
    "validate_existing_child_route_contracts",
]
