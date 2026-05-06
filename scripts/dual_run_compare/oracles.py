from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from scripts.dual_run_compare.status import (
    FALLBACK,
    GREEN,
    RED,
    SECTION_GREEN,
    SECTION_MISSING_EVIDENCE,
    SECTION_NOT_APPLICABLE,
    SECTION_PENDING_NOT_IMPLEMENTED,
    SECTION_RED,
)
from scripts.dual_run_compare.thresholds import DEFAULT_PATH


DUAL_RUN_DIR = DEFAULT_PATH.parent
NON_RAYWORKER_DIR = DUAL_RUN_DIR / "fixtures" / "non_rayworker"
REGISTRY_SNAPSHOT_PATH = NON_RAYWORKER_DIR / "registry_snapshot.json"

COMPLETION_HANDLER = "complete_task/generation-handlers.ts"
BILLING_PATH = "complete_task/billing.ts"
IDEMPOTENCY_POLICY = "complete_task_spend_ledger_idempotent_required"
REFUND_STATUS = "pending_not_implemented"
CANCELLATION_BILLING_PATH = "update-task-status/cancellationBilling.ts"
CALCULATE_TASK_COST_PATH = "calculate-task-cost/index.ts"

REQUIRED_SHARED_PATH_KEYS = frozenset(
    {
        "billing",
        "completion",
        "queue_contract",
        "shadow_side_effects",
    }
)
REQUIRED_FULL_CANARY_SECTIONS = frozenset(
    {
        "media",
        "queue_contract",
        "product_effects",
        "billing_idempotency",
        "latency",
        "vram",
        "oom",
        "error_class",
        "shadow_isolation",
    }
)
FULL_CANARY_PRODUCT_EFFECT_SECTIONS = frozenset(
    {
        "generation_vs_variant",
        "generation_created",
        "output_url_shape",
        "thumbnail_behavior",
        "parent_child_linkage",
        "shadow_visible_record_absence",
        "completion_handler",
    }
)

FULL_CANARY_PRODUCT_POLICIES: dict[str, dict[str, Any]] = {
    "video_enhance": {
        "created_as": "variant",
        "output_key": "video_url",
        "media_type": "video",
        "requires_thumbnail": True,
        "linkage": "source_variant",
    },
    "image-upscale": {
        "created_as": "variant",
        "output_key": "image_url",
        "media_type": "image",
        "requires_thumbnail": False,
        "linkage": "source_variant",
    },
    "animate_character": {
        "created_as": "generation",
        "output_key": "video_url",
        "media_type": "video",
        "requires_thumbnail": True,
        "linkage": "standalone_original",
    },
    "flux_klein_edit": {
        "created_as": "variant",
        "output_key": "image_url",
        "media_type": "image",
        "requires_thumbnail": False,
        "linkage": "source_variant",
    },
}
FULL_BILLING_IDEMPOTENCY_SECTIONS = frozenset(
    {
        "lightweight_billing_policy",
        "spend_ledger_idempotency",
        "sub_task_billing_skip",
        "video_enhance_compound_cost",
        "cancellation_cost_behavior",
        "refund_path_discovery",
    }
)


def load_registry_snapshot(path: Path = REGISTRY_SNAPSHOT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_full_canary_fixture(route_key: str, fixture_dir: Path = NON_RAYWORKER_DIR) -> dict[str, Any]:
    return json.loads((fixture_dir / f"{route_key}.json").read_text())


def active_api_routes(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        route_key: route
        for route_key, route in snapshot["routes"].items()
        if route.get("route_classification") == "active_api_owned"
    }


def worker_pool_routes(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        route_key: route
        for route_key, route in snapshot["routes"].items()
        if route.get("route_classification") == "worker_pool_fallback"
    }


def _section(key: str, status: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"key": key, "status": status}
    if reason:
        payload["reason"] = reason
    payload.update(extra)
    return payload


def _is_http_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("https://", "http://"))


def _default_product_effect_observation(route_key: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    media_extension = ".mp4" if policy["media_type"] == "video" else ".png"
    output_url = f"https://dual-run.example.invalid/{route_key}/shadow/output{media_extension}"
    thumbnail_url = f"https://dual-run.example.invalid/{route_key}/shadow/thumb.jpg" if policy["requires_thumbnail"] else None
    generation_id = f"{route_key}-generation"
    variant_id = f"{route_key}-variant"

    observation = {
        "route_key": route_key,
        "completion_handler": COMPLETION_HANDLER,
        "created_as": policy["created_as"],
        "generation_created": True,
        "generation_id": generation_id,
        "variant_id": variant_id,
        "output_key": policy["output_key"],
        "output_url": output_url,
        "thumbnail_url": thumbnail_url,
        "media_type": policy["media_type"],
        "shadow_visible_record_count": 0,
        "parent_generation_id": None,
        "child_generation_id": None,
    }
    if policy["linkage"] == "source_variant":
        observation["source_generation_id"] = generation_id
    return observation


def default_product_effect_observations() -> dict[str, dict[str, Any]]:
    return {
        route_key: _default_product_effect_observation(route_key, policy)
        for route_key, policy in FULL_CANARY_PRODUCT_POLICIES.items()
    }


def default_billing_idempotency_observations() -> dict[str, Any]:
    return {
        "spend_ledger_idempotency": {
            "source": CALCULATE_TASK_COST_PATH,
            "ledger_table": "credits_ledger",
            "ledger_type": "spend",
            "checks_existing_spend_before_insert": True,
            "skips_duplicate_spend": True,
        },
        "sub_task_billing_skip": {
            "source": "complete_task/billing.ts",
            "uses_shared_orchestrator_ref_detection": True,
            "skips_cost_trigger_for_sub_task": True,
        },
        "video_enhance_compound_cost": {
            "source": "calculate-task-cost/costHelpers.ts",
            "interpolation_compute_seconds": 10,
            "output_width": 1920,
            "output_height": 1080,
            "output_frames": 24,
            "observed_cost": 0.037883,
            "expected_cost": 0.037883,
            "breakdown_keys": ["interpolation", "upscale"],
        },
        "cancellation_cost_behavior": {
            "source": CANCELLATION_BILLING_PATH,
            "skips_child_tasks": True,
            "bills_cancelled_orchestrator_completed_children": True,
            "uses_cost_calculation_trigger": True,
            "treats_as_refund": False,
        },
        "refund_path_discovery": {
            "ledger_table": "credits_ledger",
            "ledger_type": "refund",
            "enum_allows_refund": True,
            "executable_refund_ledger_path": False,
            "source": None,
        },
    }


def _shared_path_sections(route: Mapping[str, Any]) -> list[dict[str, Any]]:
    shared_policy = route.get("shared_path_policy")
    if not isinstance(shared_policy, Mapping):
        return [
            _section(
                key,
                SECTION_MISSING_EVIDENCE,
                "shared_path_policy object missing",
                required=True,
            )
            for key in sorted(REQUIRED_SHARED_PATH_KEYS)
        ]

    sections: list[dict[str, Any]] = []
    for key in sorted(REQUIRED_SHARED_PATH_KEYS):
        policy = shared_policy.get(key)
        if policy in {"required_full", "required_lightweight"}:
            sections.append(
                _section(
                    key,
                    SECTION_GREEN,
                    required=True,
                    policy=policy,
                )
            )
        else:
            sections.append(
                _section(
                    key,
                    SECTION_RED if policy is not None else SECTION_MISSING_EVIDENCE,
                    "shared-path policy must be required_full or required_lightweight",
                    required=True,
                    policy=policy,
                )
            )
    return sections


def _fixture_sections(route_key: str, route: Mapping[str, Any], fixture_dir: Path) -> list[dict[str, Any]]:
    if route.get("canary_depth") != "full_canary":
        return [
            _section(
                "lightweight_shared_path",
                SECTION_GREEN,
                required=True,
                oracle_policy=route.get("oracle_policy"),
            )
        ]

    try:
        fixture = load_full_canary_fixture(route_key, fixture_dir)
    except FileNotFoundError:
        return [
            _section(
                "full_canary_fixture",
                SECTION_MISSING_EVIDENCE,
                "full-canary fixture is required",
                required=True,
            )
        ]

    sections: list[dict[str, Any]] = []
    required_sections = set(fixture.get("required_full_canary_sections", []))
    if required_sections == REQUIRED_FULL_CANARY_SECTIONS:
        sections.append(
            _section(
                "required_full_canary_sections",
                SECTION_GREEN,
                required=True,
                sections=sorted(required_sections),
            )
        )
    else:
        sections.append(
            _section(
                "required_full_canary_sections",
                SECTION_RED,
                "full canary fixture does not require every Sprint 3 section",
                required=True,
                missing=sorted(REQUIRED_FULL_CANARY_SECTIONS - required_sections),
                extra=sorted(required_sections - REQUIRED_FULL_CANARY_SECTIONS),
            )
        )

    for key in ("payload_shape", "output_shape"):
        value = fixture.get(key)
        sections.append(
            _section(
                key,
                SECTION_GREEN if isinstance(value, Mapping) and value else SECTION_MISSING_EVIDENCE,
                None if isinstance(value, Mapping) and value else f"{key} object missing",
                required=True,
                keys=sorted(value) if isinstance(value, Mapping) else [],
            )
        )

    for key, expected in {
        "completion_handler": COMPLETION_HANDLER,
        "billing_path": BILLING_PATH,
    }.items():
        value = fixture.get(key)
        sections.append(
            _section(
                key,
                SECTION_GREEN if value == expected else SECTION_RED,
                None if value == expected else f"{key} must be {expected}",
                required=True,
                observed=value,
                expected=expected,
            )
        )

    return sections


def _product_completion_contract_sections(route: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _section(
            "completion_contract",
            SECTION_GREEN if route.get("shared_path_policy", {}).get("completion") in {"required_full", "required_lightweight"} else SECTION_MISSING_EVIDENCE,
            None
            if route.get("shared_path_policy", {}).get("completion") in {"required_full", "required_lightweight"}
            else "completion shared-path policy missing",
            required=True,
            completion_handler=COMPLETION_HANDLER,
            policy=route.get("shared_path_policy", {}).get("completion") if isinstance(route.get("shared_path_policy"), Mapping) else None,
        )
    ]


def _product_required_section(
    key: str,
    ok: bool,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    return _section(
        key,
        SECTION_GREEN if ok else SECTION_RED,
        None if ok else reason,
        required=True,
        **extra,
    )


def _product_effect_sections(
    route_key: str,
    fixture: Mapping[str, Any],
    observation: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    policy = FULL_CANARY_PRODUCT_POLICIES.get(route_key)
    if policy is None:
        return [
            _section(
                "product_effect_policy",
                SECTION_MISSING_EVIDENCE,
                "full canary is missing a product-effect policy",
                required=True,
            )
        ]

    if observation is None:
        return [
            _section(
                key,
                SECTION_MISSING_EVIDENCE,
                "full canary product-effect observation missing",
                required=True,
            )
            for key in sorted(FULL_CANARY_PRODUCT_EFFECT_SECTIONS)
        ]

    output_shape = fixture.get("output_shape")
    output_keys = set(output_shape) if isinstance(output_shape, Mapping) else set()
    output_url = observation.get("output_url")
    thumbnail_url = observation.get("thumbnail_url")
    linkage = policy["linkage"]

    sections = [
        _product_required_section(
            "generation_vs_variant",
            observation.get("created_as") == policy["created_as"]
            and isinstance(observation.get("generation_id"), str)
            and (
                policy["created_as"] != "variant"
                or isinstance(observation.get("variant_id"), str)
            ),
            "completion asset must declare the expected generation/variant creation mode",
            observed_created_as=observation.get("created_as"),
            expected_created_as=policy["created_as"],
            generation_id=observation.get("generation_id"),
            variant_id=observation.get("variant_id"),
        ),
        _product_required_section(
            "generation_created",
            observation.get("generation_created") is True,
            "complete_task must persist generation_created=true after product record creation",
            observed=observation.get("generation_created"),
        ),
        _product_required_section(
            "output_url_shape",
            observation.get("output_key") == policy["output_key"]
            and policy["output_key"] in output_keys
            and _is_http_url(output_url),
            "completion output must use the route fixture output key and an HTTP(S) URL",
            observed_key=observation.get("output_key"),
            expected_key=policy["output_key"],
            fixture_output_keys=sorted(output_keys),
            output_url=output_url,
        ),
        _product_required_section(
            "thumbnail_behavior",
            (_is_http_url(thumbnail_url) if policy["requires_thumbnail"] else thumbnail_url is None or _is_http_url(thumbnail_url)),
            "thumbnail behavior must match route policy",
            observed_thumbnail_url=thumbnail_url,
            requires_thumbnail=policy["requires_thumbnail"],
        ),
        _product_required_section(
            "parent_child_linkage",
            (
                observation.get("source_generation_id") == observation.get("generation_id")
                if linkage == "source_variant"
                else observation.get("parent_generation_id") is None and observation.get("child_generation_id") is None
            ),
            "product record linkage must match route policy",
            linkage=linkage,
            source_generation_id=observation.get("source_generation_id"),
            parent_generation_id=observation.get("parent_generation_id"),
            child_generation_id=observation.get("child_generation_id"),
        ),
        _product_required_section(
            "shadow_visible_record_absence",
            observation.get("shadow_visible_record_count") == 0,
            "shadow mode must not create user-visible generation or variant records",
            shadow_visible_record_count=observation.get("shadow_visible_record_count"),
        ),
        _product_required_section(
            "completion_handler",
            observation.get("completion_handler") == COMPLETION_HANDLER,
            f"product effects must flow through {COMPLETION_HANDLER}",
            observed=observation.get("completion_handler"),
            expected=COMPLETION_HANDLER,
        ),
    ]
    return sections


def _billing_required_section(key: str, ok: bool, reason: str, **extra: Any) -> dict[str, Any]:
    return _section(
        key,
        SECTION_GREEN if ok else SECTION_RED,
        None if ok else reason,
        required=True,
        **extra,
    )


def _billing_policy_section(route: Mapping[str, Any]) -> dict[str, Any]:
    shared_policy = route.get("shared_path_policy")
    billing_policy = shared_policy.get("billing") if isinstance(shared_policy, Mapping) else None
    return _billing_required_section(
        "lightweight_billing_policy",
        billing_policy in {"required_full", "required_lightweight"}
        and route.get("billing_path", BILLING_PATH) == BILLING_PATH,
        "active API route must declare required billing policy and complete_task billing path",
        policy=billing_policy,
        billing_path=route.get("billing_path", BILLING_PATH),
        expected_billing_path=BILLING_PATH,
    )


def _refund_section(observation: Mapping[str, Any]) -> dict[str, Any]:
    executable = observation.get("executable_refund_ledger_path") is True
    ledger_type = observation.get("ledger_type")
    source = observation.get("source")
    if executable and ledger_type == "refund" and source != CANCELLATION_BILLING_PATH:
        return _section(
            "refund_path_discovery",
            SECTION_GREEN,
            required=True,
            ledger_table=observation.get("ledger_table"),
            ledger_type=ledger_type,
            source=source,
            executable_refund_ledger_path=True,
        )

    return _section(
        "refund_path_discovery",
        SECTION_PENDING_NOT_IMPLEMENTED,
        "no executable refund ledger path discovered; cancellation billing is not refund evidence",
        required=True,
        ledger_table=observation.get("ledger_table"),
        ledger_type=ledger_type,
        enum_allows_refund=observation.get("enum_allows_refund"),
        source=source,
        executable_refund_ledger_path=False,
    )


def _billing_idempotency_sections(
    route_key: str,
    route: Mapping[str, Any],
    observations: Mapping[str, Any],
) -> list[dict[str, Any]]:
    spend = observations.get("spend_ledger_idempotency", {})
    sub_task = observations.get("sub_task_billing_skip", {})
    video_enhance = observations.get("video_enhance_compound_cost", {})
    cancellation = observations.get("cancellation_cost_behavior", {})
    refund = observations.get("refund_path_discovery", {})

    sections = [
        _billing_policy_section(route),
        _billing_required_section(
            "spend_ledger_idempotency",
            isinstance(spend, Mapping)
            and spend.get("ledger_table") == "credits_ledger"
            and spend.get("ledger_type") == "spend"
            and spend.get("checks_existing_spend_before_insert") is True
            and spend.get("skips_duplicate_spend") is True,
            "calculate-task-cost must check existing spend ledger entries before inserting a new spend",
            source=spend.get("source") if isinstance(spend, Mapping) else None,
            ledger_table=spend.get("ledger_table") if isinstance(spend, Mapping) else None,
            ledger_type=spend.get("ledger_type") if isinstance(spend, Mapping) else None,
            checks_existing_spend_before_insert=spend.get("checks_existing_spend_before_insert") if isinstance(spend, Mapping) else None,
            skips_duplicate_spend=spend.get("skips_duplicate_spend") if isinstance(spend, Mapping) else None,
        ),
        _billing_required_section(
            "sub_task_billing_skip",
            isinstance(sub_task, Mapping)
            and sub_task.get("uses_shared_orchestrator_ref_detection") is True
            and sub_task.get("skips_cost_trigger_for_sub_task") is True,
            "complete_task billing must skip sub-task cost triggers because parent/orchestrator billing owns cost",
            source=sub_task.get("source") if isinstance(sub_task, Mapping) else None,
            uses_shared_orchestrator_ref_detection=sub_task.get("uses_shared_orchestrator_ref_detection") if isinstance(sub_task, Mapping) else None,
            skips_cost_trigger_for_sub_task=sub_task.get("skips_cost_trigger_for_sub_task") if isinstance(sub_task, Mapping) else None,
        ),
        _billing_required_section(
            "cancellation_cost_behavior",
            isinstance(cancellation, Mapping)
            and cancellation.get("source") == CANCELLATION_BILLING_PATH
            and cancellation.get("skips_child_tasks") is True
            and cancellation.get("bills_cancelled_orchestrator_completed_children") is True
            and cancellation.get("uses_cost_calculation_trigger") is True
            and cancellation.get("treats_as_refund") is False,
            "cancellation path must bill completed child work without masquerading as refund evidence",
            source=cancellation.get("source") if isinstance(cancellation, Mapping) else None,
            skips_child_tasks=cancellation.get("skips_child_tasks") if isinstance(cancellation, Mapping) else None,
            bills_cancelled_orchestrator_completed_children=cancellation.get("bills_cancelled_orchestrator_completed_children") if isinstance(cancellation, Mapping) else None,
            uses_cost_calculation_trigger=cancellation.get("uses_cost_calculation_trigger") if isinstance(cancellation, Mapping) else None,
            treats_as_refund=cancellation.get("treats_as_refund") if isinstance(cancellation, Mapping) else None,
        ),
        _refund_section(refund if isinstance(refund, Mapping) else {}),
    ]

    if route_key == "video_enhance":
        sections.append(
            _billing_required_section(
                "video_enhance_compound_cost",
                isinstance(video_enhance, Mapping)
                and video_enhance.get("source") == "calculate-task-cost/costHelpers.ts"
                and video_enhance.get("observed_cost") == video_enhance.get("expected_cost") == 0.037883
                and set(video_enhance.get("breakdown_keys", [])) == {"interpolation", "upscale"},
                "video_enhance must use compound FILM plus FlashVSR cost calculation",
                source=video_enhance.get("source") if isinstance(video_enhance, Mapping) else None,
                observed_cost=video_enhance.get("observed_cost") if isinstance(video_enhance, Mapping) else None,
                expected_cost=video_enhance.get("expected_cost") if isinstance(video_enhance, Mapping) else None,
                breakdown_keys=video_enhance.get("breakdown_keys") if isinstance(video_enhance, Mapping) else None,
            )
        )
    else:
        sections.append(
            _section(
                "video_enhance_compound_cost",
                SECTION_NOT_APPLICABLE,
                "compound video_enhance pricing only applies to video_enhance",
                required=False,
            )
        )

    return sections


def validate_shared_path_oracles(
    snapshot: Mapping[str, Any] | None = None,
    *,
    fixture_dir: Path = NON_RAYWORKER_DIR,
) -> dict[str, Any]:
    data = deepcopy(dict(snapshot)) if snapshot is not None else load_registry_snapshot()
    route_results: dict[str, dict[str, Any]] = {}

    for route_key, route in data["routes"].items():
        if route.get("route_classification") == "worker_pool_fallback":
            route_results[route_key] = {
                "route_key": route_key,
                "status": "fallback",
                "route_classification": route.get("route_classification"),
                "sections": [
                    _section(
                        "worker_pool_fallback",
                        "not_applicable",
                        "Banodoco worker-pool route is outside active API-owned oracle coverage",
                    )
                ],
            }
            continue

        sections = _shared_path_sections(route)
        sections.extend(_fixture_sections(route_key, route, fixture_dir))
        has_failure = any(section["status"] in {SECTION_RED, SECTION_MISSING_EVIDENCE} for section in sections)
        route_results[route_key] = {
            "route_key": route_key,
            "task_type": route.get("task_type", route_key),
            "status": RED if has_failure else GREEN,
            "route_classification": route.get("route_classification"),
            "canary_depth": route.get("canary_depth"),
            "oracle_policy": route.get("oracle_policy"),
            "report_status_policy": route.get("report_status_policy"),
            "completion_handler": COMPLETION_HANDLER,
            "billing_path": BILLING_PATH,
            "idempotency_policy": IDEMPOTENCY_POLICY,
            "refund_status": REFUND_STATUS,
            "sections": sections,
        }

    active_results = [route_results[route_key] for route_key in active_api_routes(data)]
    failing_active = [route["route_key"] for route in active_results if route["status"] != GREEN]
    return {
        "status": RED if failing_active else GREEN,
        "active_api_owned_route_count": len(active_results),
        "worker_pool_route_count": len(worker_pool_routes(data)),
        "failing_active_routes": failing_active,
        "routes": route_results,
    }


def validate_fixture_policy_coverage(
    snapshot: Mapping[str, Any] | None = None,
    *,
    fixture_dir: Path = NON_RAYWORKER_DIR,
) -> dict[str, Any]:
    return validate_shared_path_oracles(snapshot, fixture_dir=fixture_dir)


def validate_product_effect_oracles(
    snapshot: Mapping[str, Any] | None = None,
    *,
    fixture_dir: Path = NON_RAYWORKER_DIR,
    observations: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    data = deepcopy(dict(snapshot)) if snapshot is not None else load_registry_snapshot()
    product_observations = (
        deepcopy(dict(observations))
        if observations is not None
        else default_product_effect_observations()
    )
    route_results: dict[str, dict[str, Any]] = {}

    for route_key, route in data["routes"].items():
        if route.get("route_classification") == "worker_pool_fallback":
            route_results[route_key] = {
                "route_key": route_key,
                "status": FALLBACK,
                "route_classification": route.get("route_classification"),
                "sections": [
                    _section(
                        "worker_pool_fallback",
                        SECTION_NOT_APPLICABLE,
                        "Banodoco worker-pool route is outside active API-owned product-effect oracle coverage",
                    )
                ],
            }
            continue

        if route.get("canary_depth") != "full_canary":
            sections = _product_completion_contract_sections(route)
        else:
            try:
                fixture = load_full_canary_fixture(route_key, fixture_dir)
            except FileNotFoundError:
                fixture = {}
            sections = _product_effect_sections(route_key, fixture, product_observations.get(route_key))

        has_failure = any(section["status"] in {SECTION_RED, SECTION_MISSING_EVIDENCE} for section in sections)
        route_results[route_key] = {
            "route_key": route_key,
            "task_type": route.get("task_type", route_key),
            "status": RED if has_failure else GREEN,
            "route_classification": route.get("route_classification"),
            "canary_depth": route.get("canary_depth"),
            "oracle_policy": route.get("oracle_policy"),
            "completion_handler": COMPLETION_HANDLER,
            "required_product_effect_sections": sorted(FULL_CANARY_PRODUCT_EFFECT_SECTIONS)
            if route.get("canary_depth") == "full_canary"
            else ["completion_contract"],
            "sections": sections,
        }

    active_results = [route_results[route_key] for route_key in active_api_routes(data)]
    failing_active = [route["route_key"] for route in active_results if route["status"] != GREEN]
    return {
        "status": RED if failing_active else GREEN,
        "active_api_owned_route_count": len(active_results),
        "worker_pool_route_count": len(worker_pool_routes(data)),
        "failing_active_routes": failing_active,
        "routes": route_results,
    }


def validate_billing_idempotency_oracles(
    snapshot: Mapping[str, Any] | None = None,
    *,
    observations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data = deepcopy(dict(snapshot)) if snapshot is not None else load_registry_snapshot()
    billing_observations = (
        deepcopy(dict(observations))
        if observations is not None
        else default_billing_idempotency_observations()
    )
    route_results: dict[str, dict[str, Any]] = {}

    for route_key, route in data["routes"].items():
        if route.get("route_classification") == "worker_pool_fallback":
            route_results[route_key] = {
                "route_key": route_key,
                "status": FALLBACK,
                "route_classification": route.get("route_classification"),
                "sections": [
                    _section(
                        "worker_pool_fallback",
                        SECTION_NOT_APPLICABLE,
                        "Banodoco worker-pool route is outside active API-owned billing oracle coverage",
                    )
                ],
            }
            continue

        sections = _billing_idempotency_sections(route_key, route, billing_observations)
        has_failure = any(section["status"] in {SECTION_RED, SECTION_MISSING_EVIDENCE} for section in sections)
        pending_sections = [
            section["key"]
            for section in sections
            if section["status"] == SECTION_PENDING_NOT_IMPLEMENTED
        ]
        route_results[route_key] = {
            "route_key": route_key,
            "task_type": route.get("task_type", route_key),
            "status": RED if has_failure else GREEN,
            "route_classification": route.get("route_classification"),
            "canary_depth": route.get("canary_depth"),
            "oracle_policy": route.get("oracle_policy"),
            "billing_path": BILLING_PATH,
            "idempotency_policy": IDEMPOTENCY_POLICY,
            "refund_status": REFUND_STATUS,
            "pending_sections": pending_sections,
            "required_billing_idempotency_sections": sorted(FULL_BILLING_IDEMPOTENCY_SECTIONS)
            if route.get("canary_depth") == "full_canary"
            else [
                "lightweight_billing_policy",
                "spend_ledger_idempotency",
                "sub_task_billing_skip",
                "cancellation_cost_behavior",
                "refund_path_discovery",
            ],
            "sections": sections,
        }

    active_results = [route_results[route_key] for route_key in active_api_routes(data)]
    failing_active = [route["route_key"] for route in active_results if route["status"] != GREEN]
    pending_refund_routes = [
        route["route_key"]
        for route in active_results
        if "refund_path_discovery" in route.get("pending_sections", [])
    ]
    return {
        "status": RED if failing_active else GREEN,
        "active_api_owned_route_count": len(active_results),
        "worker_pool_route_count": len(worker_pool_routes(data)),
        "failing_active_routes": failing_active,
        "pending_refund_routes": pending_refund_routes,
        "routes": route_results,
    }
