"""Backend/profile-aware warm-cache planning for worker startup."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from source.runtime.worker.health_labels import safe_route_labels, write_warm_cache_state


DEFAULT_WARM_CACHE_MODEL_BY_ROUTE = {
    ("wgp", "1"): "wan_2_2_i2v_lightning_baseline_2_2_2",
}


@dataclass(frozen=True)
class WarmCachePlan:
    preload_model: str | None
    source: str
    skip_reason: str

    @property
    def enabled(self) -> bool:
        return bool(self.preload_model)


def resolve_warm_cache_plan(
    *,
    backend: str,
    profile: str,
    cli_preload_model: str | None = None,
    pending_tasks: bool = False,
) -> WarmCachePlan:
    if pending_tasks:
        return WarmCachePlan(preload_model=None, source="pending_task_guard", skip_reason="pending_tasks")
    if cli_preload_model:
        return WarmCachePlan(preload_model=cli_preload_model, source="cli", skip_reason="")
    direct_env = os.environ.get("REIGH_WARM_CACHE_PRELOAD_MODEL", "").strip()
    if direct_env:
        return WarmCachePlan(preload_model=direct_env, source="env", skip_reason="")

    manifest_plan = _manifest_model_for_route(backend=backend, profile=profile)
    if manifest_plan:
        return manifest_plan

    default_model = DEFAULT_WARM_CACHE_MODEL_BY_ROUTE.get((backend.lower(), str(profile)))
    if default_model:
        return WarmCachePlan(preload_model=default_model, source="default", skip_reason="")
    return WarmCachePlan(preload_model=None, source="default", skip_reason="no_matching_model")


def publish_warm_cache_state(worker_id: str, plan: WarmCachePlan, *, status: str) -> None:
    write_warm_cache_state(
        worker_id,
        {
            **safe_route_labels(),
            "warm_cache_status": status,
            "warm_cache_model": plan.preload_model or "",
            "warm_cache_source": plan.source,
            "warm_cache_skip_reason": plan.skip_reason,
            "warm_cache_updated_at": time.time(),
        },
    )


def _manifest_model_for_route(*, backend: str, profile: str) -> WarmCachePlan | None:
    data = _load_manifest()
    if not data:
        return None
    routes = data.get("routes")
    if isinstance(routes, list):
        for route in routes:
            if not isinstance(route, dict):
                continue
            if str(route.get("backend", "")).lower() != backend.lower():
                continue
            if str(route.get("profile", "")) != str(profile):
                continue
            model = str(route.get("preload_model") or "").strip()
            if model:
                return WarmCachePlan(preload_model=model, source="manifest", skip_reason="")
    by_backend = data.get(backend)
    if isinstance(by_backend, dict):
        model = str(by_backend.get(str(profile)) or by_backend.get("default") or "").strip()
        if model:
            return WarmCachePlan(preload_model=model, source="manifest", skip_reason="")
    return None


def _load_manifest() -> dict[str, Any] | None:
    inline = os.environ.get("REIGH_WARM_CACHE_CONFIG", "").strip()
    if inline:
        try:
            return json.loads(inline)
        except json.JSONDecodeError:
            return None
    manifest_path = os.environ.get("REIGH_WARM_CACHE_MANIFEST", "").strip()
    if not manifest_path:
        return None
    try:
        return json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def plan_to_metadata(plan: WarmCachePlan) -> dict[str, Any]:
    return asdict(plan)
