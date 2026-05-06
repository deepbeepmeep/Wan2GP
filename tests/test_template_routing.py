from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

httpx_stub = SimpleNamespace(
    HTTPError=Exception,
    RequestError=Exception,
    Response=object,
    TimeoutException=TimeoutError,
)
sys.modules.setdefault("httpx", httpx_stub)
sys.modules.setdefault("cv2", ModuleType("cv2"))
requests_stub = ModuleType("requests")
requests_stub.RequestException = Exception
sys.modules.setdefault("requests", requests_stub)
postgrest_exceptions = ModuleType("postgrest.exceptions")
postgrest_exceptions.APIError = Exception
sys.modules.setdefault("postgrest", ModuleType("postgrest"))
sys.modules.setdefault("postgrest.exceptions", postgrest_exceptions)


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "task_handlers"
    / "tasks"
    / "template_routing.py"
)
SELECTED_ROUTE_FIXTURES_PATH = (
    Path(__file__).resolve().parents[2]
    / "reigh-app"
    / "supabase"
    / "functions"
    / "_shared"
    / "selectedRoute.fixtures.json"
)


@pytest.fixture()
def routing():
    spec = importlib.util.spec_from_file_location("template_routing_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_supported_direct_routes_resolve_to_vibecomfy_templates(routing) -> None:
    z_image = routing.resolve_task_route(
        task_id="task-z",
        task_type="z_image_turbo",
        params={"prompt": "a small bright studio", "resolution": "1024x1024"},
        backend="vibecomfy",
    )
    assert z_image.route_key == "z_image_turbo"
    assert z_image.support_state == routing.RouteSupportState.VIBECOMFY_SUPPORTED
    assert z_image.template_id == "image/z_image"
    assert z_image.should_use_vibecomfy is True


@pytest.mark.parametrize(
    ("task_type", "expected_route_key"),
    [
        ("z_image", "z_image_turbo"),
        ("z_image_turbo_i2i", "z_image_turbo_i2i"),
        ("optimised_t2i", "wan_2_2_t2i"),
        ("qwen_image", "qwen_image"),
        ("image-upscale", "image-upscale"),
    ],
)
def test_direct_route_aliases_match_canonical_selector_keys(
    routing, task_type: str, expected_route_key: str
) -> None:
    assert routing.derive_route_key(task_type, {}) == expected_route_key


@pytest.mark.parametrize(
    "task_type",
    [
        "qwen_image_2512",
        "qwen_image",
        "qwen_image_edit",
        "qwen_image_style",
        "image_inpaint",
        "annotated_image_edit",
    ],
)
def test_sprint_qwen_direct_routes_are_explicit_wgp_only(routing, task_type: str) -> None:
    resolved = routing.resolve_task_route(
        task_id="task-unsupported",
        task_type=task_type,
        params={"prompt": "edit this"},
        backend="vibecomfy",
    )

    assert resolved.route_key == task_type
    assert resolved.support_state == routing.RouteSupportState.WGP_ONLY
    assert resolved.template_id is None
    assert resolved.should_use_vibecomfy is False
    assert resolved.fail_closed_reason
    assert "wgp_only" in resolved.fail_closed_reason
    assert "will not fall back to WGP" in resolved.fail_closed_reason


def test_routing_telemetry_fields_are_compact_and_stable(routing) -> None:
    resolved = routing.resolve_task_route(
        task_id="telemetry-task",
        task_type="z_image_turbo",
        params={
            "prompt": "telemetry",
            "resolution": "1024x1024",
            "override_profile": "3",
        },
        backend="vibecomfy",
    )

    assert routing.routing_telemetry_fields(resolved) == {
        "task_id": "telemetry-task",
        "task_type": "z_image_turbo",
        "route_key": "z_image_turbo",
        "backend": "vibecomfy",
        "template_id": "image/z_image",
        "support_state": "vibecomfy_supported",
        "memory_profile": "3",
    }


@pytest.mark.parametrize("task_type", ["travel_segment", "individual_travel_segment"])
def test_travel_children_derive_route_and_fail_closed_under_vibecomfy(
    routing, task_type: str
) -> None:
    resolved = routing.resolve_task_route(
        task_id=f"{task_type}-1",
        task_type=task_type,
        params={
            "model_name": "ltx2_distilled_19B",
            "travel_guidance": {"kind": "ltx_anchor"},
            "video_source": "/tmp/prefix.mp4",
            "override_profile": "3",
        },
        backend="vibecomfy",
    )

    assert (
        resolved.route_key
        == f"{task_type}__model-ltx2_distilled__guidance-ltx_anchor__continuity-video_source__profile-3"
    )
    assert resolved.support_state == routing.RouteSupportState.VIBECOMFY_UNSUPPORTED
    assert resolved.template_id is None
    assert resolved.should_use_vibecomfy is False
    assert resolved.fail_closed_reason
    assert "will not fall back to WGP" in resolved.fail_closed_reason


def test_template_independent_child_route_avoids_wan_vace_cocktail_keys(routing) -> None:
    resolved = routing.resolve_task_route(
        task_id="ltx-child",
        task_type="individual_travel_segment",
        params={
            "model_name": "ltx2_19B",
            "travel_mode": "ltx",
            "guidance_kind": "ltx_anchor",
            "individual_segment_params": {"base_prompt": "gentle camera drift"},
        },
        backend="vibecomfy",
    )

    combined_route_bits = " ".join(
        str(part)
        for part in (
            resolved.route_key,
            resolved.template_id,
            resolved.fail_closed_reason,
            resolved.params.get("model"),
        )
    ).lower()
    assert (
        resolved.route_key
        == "individual_travel_segment__model-ltx2__guidance-ltx_anchor__continuity-first_last__profile-default"
    )
    assert "vace" not in combined_route_bits
    assert "cocktail" not in combined_route_bits
    assert "wan_2_2" not in combined_route_bits


def test_travel_route_key_distinguishes_wan_vace_payload_context(routing) -> None:
    resolved = routing.resolve_task_route(
        task_id="wan-vace-child",
        task_type="travel_segment",
        params={
            "model_name": "wan_2_2_vace_14B",
            "travel_guidance": {"kind": "vace"},
            "video_source": "/tmp/svi-prefix.mp4",
            "wgp_profile": "default",
        },
        backend="vibecomfy",
    )

    assert (
        resolved.route_key
        == "travel_segment__model-wan22_vace__guidance-vace__continuity-video_source__profile-default"
    )
    assert resolved.support_state == routing.RouteSupportState.VIBECOMFY_UNSUPPORTED
    assert resolved.should_use_vibecomfy is False
    assert resolved.fail_closed_reason
    assert "will not fall back to WGP" in resolved.fail_closed_reason


def test_join_clips_segment_uses_dimensional_route_key(routing) -> None:
    resolved = routing.resolve_task_route(
        task_id="join-child",
        task_type="join_clips_segment",
        params={
            "model": "wan_2_2_vace_lightning_baseline_2_2_2",
            "model_family": "wan22_vace",
            "guidance_kind": "vace",
            "continuity_case": "join_bridge",
            "wgp_profile": "default",
        },
        backend="vibecomfy",
    )

    assert (
        resolved.route_key
        == "join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default"
    )
    assert resolved.support_state == routing.RouteSupportState.VIBECOMFY_UNSUPPORTED
    assert resolved.template_id is None
    assert resolved.should_use_vibecomfy is False
    assert resolved.fail_closed_reason
    assert "will not fall back to WGP" in resolved.fail_closed_reason


def test_join_clips_segment_defaults_to_join_bridge_for_wan_vace(routing) -> None:
    resolved = routing.resolve_task_route(
        task_id="join-child-defaults",
        task_type="join_clips_segment",
        params={"model": "wan_2_2_vace_lightning_baseline_2_2_2"},
        backend="wgp",
    )

    assert (
        resolved.route_key
        == "join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default"
    )
    assert resolved.fail_closed_reason is None


def test_wan_2_2_t2i_remains_wgp_only_even_when_vibecomfy_is_explicit(routing) -> None:
    resolved = routing.resolve_task_route(
        task_id="wan-image",
        task_type="wan_2_2_t2i",
        params={"prompt": "single frame", "resolution": "1024x1024"},
        backend="vibecomfy",
    )

    assert resolved.route_key == "wan_2_2_t2i"
    assert resolved.support_state == routing.RouteSupportState.WGP_ONLY
    assert resolved.should_use_vibecomfy is False
    assert resolved.fail_closed_reason
    assert "wgp_only" in resolved.fail_closed_reason


def test_unset_backend_and_wgp_backend_preserve_wgp_defaults(routing, monkeypatch) -> None:
    monkeypatch.delenv("REIGH_BACKEND", raising=False)
    unset_backend = routing.resolve_task_route(
        task_id="task-default",
        task_type="z_image_turbo",
        params={"resolution": "896x496"},
    )
    assert unset_backend.backend == routing.WorkerBackend.WGP
    assert unset_backend.fail_closed_reason is None
    assert unset_backend.should_use_vibecomfy is False

    explicit_wgp = routing.resolve_task_route(
        task_id="task-wgp",
        task_type="z_image_turbo",
        params={"resolution": "896x496"},
        backend="wgp",
    )
    assert explicit_wgp.backend == routing.WorkerBackend.WGP
    assert explicit_wgp.fail_closed_reason is None


def test_route_snapshot_fields_are_top_level_columns_plus_json_snapshot(routing) -> None:
    fields = routing.route_snapshot_fields(
        task_id="child-1",
        task_type="join_clips_segment",
        params={
            "model": "wan_2_2_vace_lightning_baseline_2_2_2",
            "model_family": "wan22_vace",
            "guidance_kind": "vace",
            "continuity_case": "join_bridge",
            "wgp_profile": "default",
        },
        backend="wgp",
        selector_namespace="production",
        selector_version=42,
        parent_route_key="join_clips_orchestrator",
    )

    route_key = (
        "join_clips_segment__model-wan22_vace__guidance-vace__"
        "continuity-join_bridge__profile-default"
    )
    assert fields == {
        "selector_namespace": "production",
        "route_key": route_key,
        "selected_backend": "wgp",
        "selector_version": 42,
        "selected_profile": "default",
        "selected_template_id": None,
        "route_run_id": None,
        "worker_contract_version": 1,
        "route_selection_snapshot": {
            "selector_namespace": "production",
            "route_key": route_key,
            "selected_backend": "wgp",
            "selector_version": 42,
            "support_state": "vibecomfy_unsupported",
            "template_id": None,
            "selected_profile": "default",
            "route_run_id": None,
            "worker_contract_version": 1,
            "task_id": "child-1",
            "parent_route_key": "join_clips_orchestrator",
        },
    }


def test_parent_derived_child_snapshot_uses_parent_contract_not_env(
    routing, monkeypatch
) -> None:
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    parent_route_contract = routing.route_snapshot_fields(
        task_id="parent-1",
        task_type="join_clips_orchestrator",
        params={},
        backend="wgp",
        selector_namespace="canary",
        selector_version=8,
        profile="production",
        run_id="run-parent-1",
    )

    fields = routing.parent_derived_child_route_snapshot_fields(
        parent_params={"route_contract": parent_route_contract},
        child_task_id="child-1",
        child_task_type="join_clips_segment",
        child_params={
            "model": "wan_2_2_vace_lightning_baseline_2_2_2",
            "model_family": "wan22_vace",
            "guidance_kind": "vace",
            "continuity_case": "join_bridge",
            "wgp_profile": "default",
        },
    )

    route_key = (
        "join_clips_segment__model-wan22_vace__guidance-vace__"
        "continuity-join_bridge__profile-default"
    )
    assert fields["route_key"] == route_key
    assert fields["selected_backend"] == "wgp"
    assert fields["selector_namespace"] == "canary"
    assert fields["selector_version"] == 8
    assert fields["selected_profile"] == "production"
    assert fields["route_run_id"] == "run-parent-1"
    assert fields["route_selection_snapshot"]["parent_route_key"] == "join_clips_orchestrator"
    assert fields["route_selection_snapshot"]["task_id"] == "child-1"


def test_parent_child_route_preflight_reports_incompatible_vibecomfy_child(routing) -> None:
    parent_route_contract = routing.route_snapshot_fields(
        task_id="parent-1",
        task_type="join_clips_orchestrator",
        params={},
        backend="vibecomfy",
        selector_namespace="canary",
        selector_version=8,
        profile="production",
    )

    preflight = routing.preflight_parent_child_route(
        parent_params={"route_contract": parent_route_contract},
        child_task_type="join_clips_segment",
        child_params={
            "model": "wan_2_2_vace_lightning_baseline_2_2_2",
            "model_family": "wan22_vace",
            "guidance_kind": "vace",
            "continuity_case": "join_bridge",
        },
    )

    assert preflight.ok is False
    assert preflight.parent_route_key == "join_clips_orchestrator"
    assert preflight.child_route_key == (
        "join_clips_segment__model-wan22_vace__guidance-vace__"
        "continuity-join_bridge__profile-default"
    )
    assert preflight.fail_closed_reason
    assert "will not fall back to WGP" in preflight.fail_closed_reason


def test_parent_derived_child_snapshot_fails_closed_on_missing_parent_contract(
    routing,
) -> None:
    with pytest.raises(ValueError, match="Missing or malformed parent params.route_contract"):
        routing.parent_derived_child_route_snapshot_fields(
            parent_params={},
            child_task_type="join_clips_segment",
            child_params={},
        )


def test_parent_derived_child_snapshot_fails_closed_on_malformed_parent_contract(
    routing,
) -> None:
    with pytest.raises(ValueError, match="selected_backend is invalid"):
        routing.parent_derived_child_route_snapshot_fields(
            parent_params={
                "route_contract": {
                    "route_key": "join_clips_orchestrator",
                    "selected_backend": "comfy",
                    "selector_namespace": "canary",
                    "selected_profile": "production",
                    "worker_contract_version": 1,
                }
            },
            child_task_type="join_clips_segment",
            child_params={},
        )


def test_parent_derived_child_snapshot_fails_closed_for_vibecomfy_blocked_child(
    routing,
) -> None:
    parent_route_contract = routing.route_snapshot_fields(
        task_id="parent-1",
        task_type="travel_orchestrator",
        params={},
        backend="vibecomfy",
        selector_namespace="canary",
        selector_version=8,
        profile="production",
    )

    with pytest.raises(ValueError, match="explicit VibeComfy backend will not fall back to WGP"):
        routing.parent_derived_child_route_snapshot_fields(
            parent_params={"route_contract": parent_route_contract},
            child_task_type="travel_stitch",
            child_params={},
        )


@pytest.mark.parametrize(
    ("parent_task_type", "child_task_type", "child_params", "expected_parent_route_key"),
    [
        ("travel_orchestrator", "travel_stitch", {}, "travel_orchestrator"),
        (
            "travel_orchestrator",
            "join_clips_orchestrator",
            {"orchestrator_details": {"use_parallel_joins": True}},
            "travel_orchestrator",
        ),
        ("join_clips_orchestrator", "join_final_stitch", {}, "join_clips_orchestrator"),
        ("edit_video_orchestrator", "join_final_stitch", {}, "edit_video_orchestrator"),
    ],
)
def test_parent_derived_control_rows_are_classified_and_parent_stamped(
    routing,
    parent_task_type: str,
    child_task_type: str,
    child_params: dict,
    expected_parent_route_key: str,
) -> None:
    parent_route_contract = routing.route_snapshot_fields(
        task_id=f"{parent_task_type}-1",
        task_type=parent_task_type,
        params={},
        backend="wgp",
        selector_namespace="canary",
        selector_version=12,
        profile="production",
        run_id="run-control-1",
    )

    fields = routing.parent_derived_child_route_snapshot_fields(
        parent_params={"route_contract": parent_route_contract},
        child_task_type=child_task_type,
        child_params=child_params,
    )

    assert fields["route_key"] == child_task_type
    assert fields["selected_backend"] == "wgp"
    assert fields["selector_namespace"] == "canary"
    assert fields["selector_version"] == 12
    assert fields["selected_profile"] == "production"
    assert fields["route_run_id"] == "run-control-1"
    assert fields["route_selection_snapshot"]["support_state"] == "wgp_only"
    assert fields["route_selection_snapshot"]["parent_route_key"] == expected_parent_route_key


@pytest.mark.parametrize("child_task_type", ["travel_stitch", "join_clips_orchestrator", "join_final_stitch"])
def test_parent_derived_control_rows_require_parent_route_contract(
    routing, child_task_type: str
) -> None:
    with pytest.raises(ValueError, match="Missing or malformed parent params.route_contract"):
        routing.parent_derived_child_route_snapshot_fields(
            parent_params={},
            child_task_type=child_task_type,
            child_params={"orchestrator_details": {}},
        )


@pytest.mark.parametrize(
    ("mutator", "expected_reason"),
    [
        (lambda contract: contract.update({"selected_backend": "vibecomfy"}), "selected_backend"),
        (lambda contract: contract.update({"selector_version": 99}), "selector_version"),
        (lambda contract: contract.update({"selected_profile": "3"}), "selected_profile"),
        (
            lambda contract: contract["route_selection_snapshot"].update(
                {"parent_route_key": "edit_video_orchestrator"}
            ),
            "parent_route_key",
        ),
    ],
)
def test_existing_child_route_contract_consistency_detects_mixed_route_identity(
    routing, mutator, expected_reason: str
) -> None:
    parent_route_contract = routing.route_snapshot_fields(
        task_id="join-parent-1",
        task_type="join_clips_orchestrator",
        params={},
        backend="wgp",
        selector_namespace="canary",
        selector_version=12,
        profile="production",
    )
    child_route_contract = routing.parent_derived_child_route_snapshot_fields(
        parent_params={"route_contract": parent_route_contract},
        child_task_type="join_final_stitch",
        child_params={},
    )
    mutator(child_route_contract)

    result = routing.validate_existing_child_route_contracts(
        parent_params={"route_contract": parent_route_contract},
        child_tasks=[{"id": "child-1", "params": {"route_contract": child_route_contract}}],
        expected_parent_route_key="join_clips_orchestrator",
    )

    assert result.ok is False
    assert result.mismatched_task_ids == ("child-1",)
    assert result.fail_closed_reason
    assert expected_reason in result.fail_closed_reason


def test_existing_child_route_contract_consistency_accepts_matching_children(
    routing,
) -> None:
    parent_route_contract = routing.route_snapshot_fields(
        task_id="travel-parent-1",
        task_type="travel_orchestrator",
        params={},
        backend="wgp",
        selector_namespace="canary",
        selector_version=12,
        profile="production",
    )
    child_route_contract = routing.parent_derived_child_route_snapshot_fields(
        parent_params={"route_contract": parent_route_contract},
        child_task_type="travel_stitch",
        child_params={},
    )

    result = routing.validate_existing_child_route_contracts(
        parent_params={"route_contract": parent_route_contract},
        child_tasks=[
            {
                "id": "stitch-1",
                "params": json.dumps({"route_contract": child_route_contract}),
            }
        ],
        expected_parent_route_key="travel_orchestrator",
    )

    assert result.ok is True
    assert result.fail_closed_reason is None
    assert result.mismatched_task_ids == ()


def test_existing_child_route_contract_consistency_fails_closed_on_missing_contract(
    routing,
) -> None:
    parent_route_contract = routing.route_snapshot_fields(
        task_id="travel-parent-1",
        task_type="travel_orchestrator",
        params={},
        backend="wgp",
        selector_namespace="canary",
        selector_version=12,
        profile="production",
    )

    result = routing.validate_existing_child_route_contracts(
        parent_params={"route_contract": parent_route_contract},
        child_tasks=[{"id": "segment-1", "params": {"segment_index": 0}}],
        expected_parent_route_key="travel_orchestrator",
    )

    assert result.ok is False
    assert result.mismatched_task_ids == ("segment-1",)
    assert result.fail_closed_reason
    assert "missing params.route_contract" in result.fail_closed_reason


def test_python_route_contract_matches_edge_fixtures(routing) -> None:
    fixtures = json.loads(SELECTED_ROUTE_FIXTURES_PATH.read_text(encoding="utf-8"))

    for fixture in fixtures:
        input_payload = fixture["input"]
        expected = fixture["expected"]
        assert routing.route_snapshot_fields(
            task_type=input_payload["task_type"],
            params=input_payload.get("params"),
            task_id=input_payload.get("task_id"),
            backend=input_payload.get("backend"),
            selector_namespace=input_payload.get("selector_namespace", "production"),
            selector_version=input_payload.get("selector_version"),
            parent_route_key=input_payload.get("parent_route_key"),
            profile=input_payload.get("profile"),
            run_id=input_payload.get("run_id"),
            worker_contract_version=input_payload.get("worker_contract_version", 1),
        ) == expected, fixture["name"]


def test_reigh_backend_vibecomfy_is_parsed_strictly(routing, monkeypatch) -> None:
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    resolved = routing.resolve_task_route(
        task_id="task-env",
        task_type="z_image_turbo",
        params={"prompt": "env-selected"},
    )
    assert resolved.backend == routing.WorkerBackend.VIBECOMFY
    assert resolved.should_use_vibecomfy is True

    with pytest.raises(ValueError, match="Unsupported REIGH_BACKEND"):
        routing.parse_worker_backend("comfy")


def test_z_image_non_default_resolution_remains_vibecomfy_supported(routing) -> None:
    resolved = routing.resolve_task_route(
        task_id="z-image-non-default",
        task_type="z_image_turbo",
        params={"prompt": "non default", "resolution": "896x496"},
        backend="vibecomfy",
    )

    assert resolved.should_use_vibecomfy is True
    assert resolved.fail_closed_reason is None


def test_module_has_no_wgp_heavy_or_vibecomfy_imports() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

    imported_roots: set[str] = set()
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
            imported_roots.add(node.module.split(".", 1)[0])

    assert "vibecomfy" not in imported_roots
    assert "source" not in imported_roots
    assert not any("wgp" in module.lower() for module in imported_modules)


def test_claim_route_guard_requeues_backend_mismatch(monkeypatch) -> None:
    from source.core.db import task_claim

    requeue_calls = []

    def _fake_requeue(task_id, error_message, current_attempts, error_category=None):
        requeue_calls.append((task_id, error_message, current_attempts, error_category))
        return True

    monkeypatch.setattr(
        "source.core.db.lifecycle.task_status_retry.requeue_task_for_retry",
        _fake_requeue,
    )

    allowed = task_claim._claim_route_guard(
        {
            "task_id": "task-mismatch",
            "attempts": 2,
            "claimed_backend": "vibecomfy",
            "claimed_selector_namespace": "production",
            "claimed_route_key": "z_image_turbo",
            "claim_decision_reason": "eligible",
        },
        task_claim.WorkerBackend.WGP,
    )

    assert allowed is False
    assert len(requeue_calls) == 1
    assert requeue_calls[0][0] == "task-mismatch"
    assert requeue_calls[0][2] == 2
    assert requeue_calls[0][3] == "backend_mismatch"


def test_claim_route_guard_fails_closed_on_malformed_decision(monkeypatch) -> None:
    from source.core.db import task_claim

    failed_calls = []
    requeue_calls = []

    def _fake_fail(task_id, message):
        failed_calls.append((task_id, message))
        return True

    def _fake_requeue(task_id, error_message, current_attempts, error_category=None):
        requeue_calls.append((task_id, error_message, current_attempts, error_category))
        return True

    monkeypatch.setattr(
        "source.core.db.lifecycle.task_status_complete_remote.mark_task_failed_via_edge_function",
        _fake_fail,
    )
    monkeypatch.setattr(
        "source.core.db.lifecycle.task_status_retry.requeue_task_for_retry",
        _fake_requeue,
    )

    assert task_claim._claim_route_guard(
        {
            "task_id": "task-malformed",
            "attempts": 4,
            "claimed_backend": "comfy",
            "claimed_selector_namespace": "production",
            "claimed_route_key": "z_image_turbo",
            "claim_decision_reason": "eligible",
        },
        task_claim.WorkerBackend.WGP,
    ) is False
    assert len(failed_calls) == 1
    assert failed_calls[0][0] == "task-malformed"
    assert "Malformed route/backend claim decision before execution" in failed_calls[0][1]
    assert requeue_calls == []


def test_claim_route_guard_requeues_when_fail_closed_update_is_unavailable(monkeypatch) -> None:
    from source.core.db import task_claim

    failed_calls = []
    requeue_calls = []

    def _fake_fail(task_id, message):
        failed_calls.append((task_id, message))
        return False

    def _fake_requeue(task_id, error_message, current_attempts, error_category=None):
        requeue_calls.append((task_id, error_message, current_attempts, error_category))
        return True

    monkeypatch.setattr(
        "source.core.db.lifecycle.task_status_complete_remote.mark_task_failed_via_edge_function",
        _fake_fail,
    )
    monkeypatch.setattr(
        "source.core.db.lifecycle.task_status_retry.requeue_task_for_retry",
        _fake_requeue,
    )

    assert task_claim._claim_route_guard(
        {
            "task_id": "task-unsupported",
            "attempts": 1,
            "claimed_backend": "wgp",
            "claimed_selector_namespace": "production",
            "claimed_route_key": "z_image_turbo",
            "claim_decision_reason": "unsupported_capability",
        },
        task_claim.WorkerBackend.WGP,
    ) is False
    assert len(failed_calls) == 1
    assert "Unsupported claim decision reason before execution" in failed_calls[0][1]
    assert len(requeue_calls) == 1
    assert requeue_calls[0][0] == "task-unsupported"
    assert requeue_calls[0][2] == 1
    assert requeue_calls[0][3] == "route_decision_fail_closed"


def test_claim_flow_sends_backend_and_selector_namespace(monkeypatch) -> None:
    from source.core.db.config import DBRuntimeConfig
    from source.core.db.lifecycle.task_claim_flow import (
        TaskClaimFlowDependencies,
        claim_oldest_queued_task,
    )

    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    monkeypatch.setenv("REIGH_SELECTOR_NAMESPACE", "staging")
    monkeypatch.setenv("REIGH_SELECTOR_VERSION", "42")
    monkeypatch.setenv("REIGH_WORKER_PROFILE", "3")
    monkeypatch.setenv("REIGH_WORKER_CONTRACT_VERSION", "7")
    captured_payload = {}

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"task_id": "claimed-2", "task_type": "z_image_turbo", "params": {}}

    def _call_edge_function_with_retry(**kwargs):
        captured_payload.update(kwargs["payload"])
        return _Response(), None

    runtime = DBRuntimeConfig(
        db_type="supabase",
        pg_table_name="tasks",
        supabase_url="https://example.supabase.co",
        supabase_service_key="svc",
        supabase_video_bucket="bucket",
        supabase_client=object(),
        supabase_access_token="token",
        supabase_edge_complete_task_url=None,
        supabase_edge_create_task_url=None,
        supabase_edge_claim_task_url="https://example.supabase.co/functions/v1/claim-next-task",
        debug_mode=False,
    )
    result = claim_oldest_queued_task(
        worker_id="worker-vibe",
        runtime=runtime,
        deps=TaskClaimFlowDependencies(
            check_my_assigned_tasks=lambda *_args, **_kwargs: None,
            check_task_counts_supabase=lambda *_args, **_kwargs: {"totals": {"queued_only": 1}},
            orchestrator_has_incomplete_children=lambda *_args, **_kwargs: False,
            register_orchestrator_deferral=lambda _task_id: (False, 1),
            clear_orchestrator_deferral=lambda _task_id: None,
            resolve_edge_request=lambda *_args, **_kwargs: SimpleNamespace(
                url="https://example.supabase.co/functions/v1/claim-next-task",
                headers={"Authorization": "Bearer token"},
            ),
            call_edge_function_with_retry=_call_edge_function_with_retry,
            has_required_edge_credentials=lambda headers: "Authorization" in headers,
        ),
    )

    assert result == {"task_id": "claimed-2", "task_type": "z_image_turbo", "params": {}}
    assert captured_payload == {
        "worker_id": "worker-vibe",
        "run_type": "gpu",
        "worker_backend": "vibecomfy",
        "worker_profile": "3",
        "selector_namespace": "staging",
        "selector_version": "42",
        "worker_contract_version": 7,
    }


def test_direct_claim_payload_sends_backend_profile_selector_and_contract(monkeypatch) -> None:
    from source.core.db import task_claim

    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    monkeypatch.setenv("REIGH_SELECTOR_NAMESPACE", "canary")
    monkeypatch.setenv("REIGH_SELECTOR_VERSION", "43")
    monkeypatch.setenv("REIGH_WORKER_PROFILE", "3")
    monkeypatch.setenv("REIGH_WORKER_CONTRACT_VERSION", "8")
    monkeypatch.setattr(task_claim._cfg, "SUPABASE_URL", "https://example.supabase.co", raising=False)
    monkeypatch.setattr(task_claim._cfg, "SUPABASE_ACCESS_TOKEN", "token", raising=False)
    monkeypatch.setattr(
        task_claim._cfg,
        "SUPABASE_EDGE_CLAIM_TASK_URL",
        "https://example.supabase.co/functions/v1/claim-next-task",
        raising=False,
    )
    captured_payloads = []

    class _Response:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self._body = body
            self.text = json.dumps(body)

        def json(self):
            return self._body

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured_payloads.append((url, dict(json or {})))
        if url.endswith("/task-counts"):
            return _Response(200, {"totals": {"queued_only": 1, "eligible_queued": 1}})
        return _Response(204, {})

    monkeypatch.setattr(task_claim.httpx, "post", _fake_post, raising=False)

    outcome, task = task_claim.poll_next_task(
        worker_id="worker-direct-vibe",
        same_model_only=True,
        max_task_wait_minutes=9,
    )

    assert outcome == task_claim.ClaimPollOutcome.EMPTY
    assert task is None
    task_counts_payload = captured_payloads[0][1]
    claim_payload = captured_payloads[1][1]
    for payload in (task_counts_payload, claim_payload):
        assert payload["worker_backend"] == "vibecomfy"
        assert payload["worker_profile"] == "3"
        assert payload["selector_namespace"] == "canary"
        assert payload["selector_version"] == "43"
        assert payload["worker_contract_version"] == 8
    assert claim_payload["worker_id"] == "worker-direct-vibe"
    assert claim_payload["same_model_only"] is True
    assert claim_payload["max_task_wait_minutes"] == 9


def test_direct_claim_suppresses_edge_call_when_disk_near_full(monkeypatch) -> None:
    from source.core.db import task_claim
    from source.runtime.worker.resource_pressure import ResourcePressureResult
    import source.runtime.worker.resource_pressure as resource_pressure

    monkeypatch.setattr(task_claim._cfg, "SUPABASE_URL", "https://example.supabase.co", raising=False)
    monkeypatch.setattr(task_claim._cfg, "SUPABASE_ACCESS_TOKEN", "token", raising=False)
    monkeypatch.setattr(
        task_claim._cfg,
        "SUPABASE_EDGE_CLAIM_TASK_URL",
        "https://example.supabase.co/functions/v1/claim-next-task",
        raising=False,
    )
    monkeypatch.setattr(
        resource_pressure,
        "ensure_resources_for_claim",
        lambda _worker_id: ResourcePressureResult(
            status="near_full",
            action="claim_suppressed",
            allow_work=False,
            quota_alert=True,
            required_free_bytes=1024,
            recovered_bytes=0,
            volumes=(),
            cleanup={"lora": {}, "artifacts": {}},
            reason="disk_pressure_unrecoverable",
        ),
    )
    calls = []

    def _fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("claim path should not call edge while disk is near full")

    monkeypatch.setattr(task_claim.httpx, "post", _fake_post, raising=False)

    outcome, task = task_claim.poll_next_task(
        worker_id="worker-disk-full",
        same_model_only=True,
        max_task_wait_minutes=9,
    )

    assert outcome == task_claim.ClaimPollOutcome.EMPTY
    assert task is None
    assert calls == []


def test_add_task_to_db_lifts_route_snapshot_fields_into_create_task_payload(monkeypatch) -> None:
    from source.core.db import task_completion

    captured_payload = {}

    class _Response:
        status_code = 200
        text = "ok"

    def _fake_edge_call(**kwargs):
        captured_payload.update(kwargs["payload"])
        return _Response(), None

    monkeypatch.setattr(task_completion._cfg, "SUPABASE_EDGE_CREATE_TASK_URL", "https://edge.test/create-task", raising=False)
    monkeypatch.setattr(task_completion._cfg, "SUPABASE_URL", None, raising=False)
    monkeypatch.setattr(task_completion._cfg, "SUPABASE_ACCESS_TOKEN", "token", raising=False)
    monkeypatch.setattr(task_completion, "_call_edge_function_with_retry", _fake_edge_call)
    monkeypatch.setattr("uuid.uuid4", lambda: "task-fixed-id")

    route_fields = {
        "selector_namespace": "production",
        "route_key": "join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default",
        "selected_backend": "wgp",
        "selector_version": 12,
        "selected_profile": "default",
        "selected_template_id": None,
        "route_run_id": "run-1",
        "worker_contract_version": 1,
        "route_selection_snapshot": {
            "selector_namespace": "production",
            "route_key": "join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default",
            "selected_backend": "wgp",
            "selector_version": 12,
            "selected_profile": "default",
            "route_run_id": "run-1",
            "worker_contract_version": 1,
        },
    }

    task_id = task_completion.add_task_to_db(
        task_payload={"project_id": "project-1", "prompt": "bridge"},
        task_type_str="join_clips_segment",
        route_snapshot_fields=route_fields,
    )

    assert task_id == "task-fixed-id"
    assert captured_payload["input"]["task_id"] == "task-fixed-id"
    for key, value in route_fields.items():
        assert captured_payload["input"][key] == value
    assert captured_payload["input"]["prompt"] == "bridge"


@pytest.mark.parametrize(
    ("dependant_on", "expected_dependant_on"),
    [
        ("created-1", ["created-1"]),
        (["created-1", "created-2"], ["created-1", "created-2"]),
    ],
)
def test_add_task_to_db_preserves_control_row_dependency_arrays(
    monkeypatch, dependant_on, expected_dependant_on
) -> None:
    from source.core.db import task_completion

    captured_payload = {}

    class _Response:
        status_code = 200
        text = "ok"

    def _fake_edge_call(**kwargs):
        captured_payload.update(kwargs["payload"])
        return _Response(), None

    monkeypatch.setattr(
        task_completion._cfg,
        "SUPABASE_EDGE_CREATE_TASK_URL",
        "https://edge.test/create-task",
        raising=False,
    )
    monkeypatch.setattr(task_completion._cfg, "SUPABASE_URL", None, raising=False)
    monkeypatch.setattr(task_completion._cfg, "SUPABASE_ACCESS_TOKEN", "token", raising=False)
    monkeypatch.setattr(task_completion, "_call_edge_function_with_retry", _fake_edge_call)
    monkeypatch.setattr("uuid.uuid4", lambda: "final-stitch-fixed-id")

    route_fields = {
        "selector_namespace": "canary",
        "route_key": "join_final_stitch",
        "selected_backend": "wgp",
        "selector_version": 12,
        "selected_profile": "production",
        "selected_template_id": None,
        "route_run_id": "run-1",
        "worker_contract_version": 1,
        "route_selection_snapshot": {
            "selector_namespace": "canary",
            "route_key": "join_final_stitch",
            "selected_backend": "wgp",
            "selector_version": 12,
            "support_state": "wgp_only",
            "selected_profile": "production",
            "route_run_id": "run-1",
            "worker_contract_version": 1,
            "parent_route_key": "join_clips_orchestrator",
        },
    }

    task_id = task_completion.add_task_to_db(
        task_payload={"project_id": "project-1", "transition_task_ids": expected_dependant_on},
        task_type_str="join_final_stitch",
        dependant_on=dependant_on,
        route_snapshot_fields=route_fields,
    )

    assert task_id == "final-stitch-fixed-id"
    assert captured_payload["input"]["dependant_on"] == expected_dependant_on
    assert captured_payload["input"]["route_key"] == "join_final_stitch"
    assert captured_payload["input"]["route_selection_snapshot"]["parent_route_key"] == "join_clips_orchestrator"


def test_parallel_join_child_creation_derives_transition_and_final_snapshots_from_parent(
    routing, monkeypatch, tmp_path
) -> None:
    from source.task_handlers.join import task_builder

    calls = []

    def _fake_add_task_to_db(**kwargs):
        calls.append(kwargs)
        return f"created-{len(calls)}"

    monkeypatch.setattr(task_builder, "add_task_to_db", _fake_add_task_to_db)
    parent_route_contract = routing.route_snapshot_fields(
        task_id="join-orch-1",
        task_type="join_clips_orchestrator",
        params={},
        backend="wgp",
        selector_namespace="canary",
        selector_version=8,
        profile="production",
        run_id="run-parent-1",
    )

    ok, message = task_builder._create_parallel_join_tasks(
        clip_list=[
            {"url": "https://example.com/a.mp4"},
            {"url": "https://example.com/b.mp4"},
            {"url": "https://example.com/c.mp4"},
        ],
        run_id="join-run-1",
        join_settings={
            "model": "wan_2_2_vace_lightning_baseline_2_2_2",
            "guidance_kind": "vace",
            "continuity_case": "join_bridge",
            "wgp_profile": "default",
            "fps": 16,
        },
        per_join_settings=[],
        vlm_enhanced_prompts=[],
        current_run_output_dir=tmp_path,
        orchestrator_task_id_str="join-orch-1",
        orchestrator_project_id="project-1",
        orchestrator_payload={"fps": 16},
        parent_generation_id="parent-1",
        parent_params={"route_contract": parent_route_contract},
        parent_route_key="join_clips_orchestrator",
    )

    assert ok is True
    assert "parallel transitions" in message
    assert [call["task_type_str"] for call in calls] == [
        "join_clips_segment",
        "join_clips_segment",
        "join_final_stitch",
    ]
    assert calls[2]["dependant_on"] == ["created-1", "created-2"]
    for call in calls[:2]:
        assert call["route_snapshot_fields"]["route_key"] == (
            "join_clips_segment__model-wan22_vace__guidance-vace__"
            "continuity-join_bridge__profile-default"
        )
        assert call["route_snapshot_fields"]["selected_backend"] == "wgp"
        assert call["route_snapshot_fields"]["selector_namespace"] == "canary"
        assert call["route_snapshot_fields"]["selector_version"] == 8
        assert call["route_snapshot_fields"]["selected_profile"] == "production"
        assert call["route_snapshot_fields"]["route_selection_snapshot"]["parent_route_key"] == "join_clips_orchestrator"
    assert calls[2]["route_snapshot_fields"]["route_key"] == "join_final_stitch"
    assert calls[2]["route_snapshot_fields"]["selected_backend"] == "wgp"
    assert calls[2]["route_snapshot_fields"]["selector_namespace"] == "canary"
    assert calls[2]["route_snapshot_fields"]["route_selection_snapshot"]["parent_route_key"] == "join_clips_orchestrator"


def test_edit_video_join_child_creation_uses_edit_video_parent_route_key(
    routing, monkeypatch, tmp_path
) -> None:
    from source.task_handlers.join import task_builder

    calls = []

    def _fake_add_task_to_db(**kwargs):
        calls.append(kwargs)
        return f"created-{len(calls)}"

    monkeypatch.setattr(task_builder, "add_task_to_db", _fake_add_task_to_db)
    parent_route_contract = routing.route_snapshot_fields(
        task_id="edit-video-orch-1",
        task_type="edit_video_orchestrator",
        params={},
        backend="wgp",
        selector_namespace="canary",
        selector_version=11,
        profile="production",
        run_id="run-edit-parent-1",
    )

    ok, message = task_builder._create_join_chain_tasks(
        clip_list=[
            {"url": "https://example.com/keeper-a.mp4"},
            {"url": "https://example.com/keeper-b.mp4"},
            {"url": "https://example.com/keeper-c.mp4"},
        ],
        run_id="edit-video-run-1",
        join_settings={
            "model": "wan_2_2_vace_lightning_baseline_2_2_2",
            "guidance_kind": "vace",
            "continuity_case": "join_bridge",
            "wgp_profile": "default",
            "fps": 24,
        },
        per_join_settings=[],
        vlm_enhanced_prompts=[],
        current_run_output_dir=tmp_path,
        orchestrator_task_id_str="edit-video-orch-1",
        orchestrator_project_id="project-1",
        orchestrator_payload={"fps": 24},
        parent_generation_id="parent-edit-1",
        parent_params={"route_contract": parent_route_contract},
        parent_route_key="edit_video_orchestrator",
    )

    assert ok is True
    assert "chain joins" in message
    assert [call["task_type_str"] for call in calls] == [
        "join_clips_segment",
        "join_clips_segment",
        "join_final_stitch",
    ]
    assert calls[0]["dependant_on"] is None
    assert calls[1]["dependant_on"] == "created-1"
    assert calls[2]["dependant_on"] == "created-2"
    for call in calls:
        snapshot = call["route_snapshot_fields"]["route_selection_snapshot"]
        assert snapshot["parent_route_key"] == "edit_video_orchestrator"
        assert call["route_snapshot_fields"]["selector_namespace"] == "canary"
        assert call["route_snapshot_fields"]["selector_version"] == 11
        assert call["route_snapshot_fields"]["selected_profile"] == "production"
    assert calls[2]["route_snapshot_fields"]["route_key"] == "join_final_stitch"


def test_join_child_creation_fails_closed_when_parent_route_identity_mismatches(
    routing, monkeypatch, tmp_path
) -> None:
    from source.task_handlers.join import task_builder

    calls = []

    def _fake_add_task_to_db(**kwargs):
        calls.append(kwargs)
        return f"created-{len(calls)}"

    monkeypatch.setattr(task_builder, "add_task_to_db", _fake_add_task_to_db)
    parent_route_contract = routing.route_snapshot_fields(
        task_id="edit-video-orch-2",
        task_type="edit_video_orchestrator",
        params={},
        backend="wgp",
        selector_namespace="canary",
        selector_version=11,
        profile="production",
    )

    with pytest.raises(ValueError, match="Parent route key mismatch"):
        task_builder._create_join_chain_tasks(
            clip_list=[
                {"url": "https://example.com/keeper-a.mp4"},
                {"url": "https://example.com/keeper-b.mp4"},
            ],
            run_id="edit-video-run-2",
            join_settings={
                "model": "wan_2_2_vace_lightning_baseline_2_2_2",
                "guidance_kind": "vace",
                "continuity_case": "join_bridge",
                "wgp_profile": "default",
            },
            per_join_settings=[],
            vlm_enhanced_prompts=[],
            current_run_output_dir=tmp_path,
            orchestrator_task_id_str="edit-video-orch-2",
            orchestrator_project_id="project-1",
            orchestrator_payload={"fps": 24},
            parent_generation_id="parent-edit-2",
            parent_params={"route_contract": parent_route_contract},
            parent_route_key="join_clips_orchestrator",
        )

    assert calls == []


def test_parallel_join_child_creation_falls_back_only_without_parent_contract(
    monkeypatch, tmp_path
) -> None:
    from source.task_handlers.join import task_builder

    calls = []

    def _fake_add_task_to_db(**kwargs):
        calls.append(kwargs)
        return f"created-{len(calls)}"

    monkeypatch.setattr(task_builder, "add_task_to_db", _fake_add_task_to_db)

    ok, message = task_builder._create_parallel_join_tasks(
        clip_list=[
            {"url": "https://example.com/a.mp4"},
            {"url": "https://example.com/b.mp4"},
        ],
        run_id="join-run-direct",
        join_settings={
            "model": "wan_2_2_vace_lightning_baseline_2_2_2",
            "model_family": "wan22_vace",
            "guidance_kind": "vace",
            "continuity_case": "join_bridge",
            "wgp_profile": "default",
        },
        per_join_settings=[],
        vlm_enhanced_prompts=[],
        current_run_output_dir=tmp_path,
        orchestrator_task_id_str="join-orch-direct",
        orchestrator_project_id="project-1",
        orchestrator_payload={"fps": 16},
        parent_generation_id="parent-1",
    )

    assert ok is True
    assert "parallel transitions" in message
    assert calls[0]["route_snapshot_fields"]["selected_backend"] == "wgp"
    assert calls[0]["route_snapshot_fields"]["selector_namespace"] == "production"
    assert calls[0]["route_snapshot_fields"]["route_selection_snapshot"]["parent_route_key"] == "join_clips_orchestrator"


@pytest.mark.parametrize(
    ("builder_name", "expected_cancel_context"),
    [
        ("_create_join_chain_tasks", "aborting join creation at index 1"),
        ("_create_parallel_join_tasks", "aborting transition creation at index 1"),
    ],
)
def test_join_child_creation_cancellation_mid_spawn_preserves_existing_children(
    routing,
    monkeypatch,
    tmp_path,
    builder_name: str,
    expected_cancel_context: str,
) -> None:
    from source.task_handlers.join import shared as join_shared
    from source.task_handlers.join import task_builder

    calls = []
    cancelled = []
    status_checks = iter(["In Progress", "Cancelled"])

    def _fake_add_task_to_db(**kwargs):
        calls.append(kwargs)
        return f"created-{len(calls)}"

    monkeypatch.setattr(task_builder, "add_task_to_db", _fake_add_task_to_db)
    monkeypatch.setattr(join_shared, "get_task_current_status", lambda _task_id: next(status_checks))
    monkeypatch.setattr(
        join_shared,
        "cancel_orchestrator_children",
        lambda task_id, reason: cancelled.append((task_id, reason)),
    )

    parent_route_contract = routing.route_snapshot_fields(
        task_id="join-orch-cancel",
        task_type="join_clips_orchestrator",
        params={},
        backend="wgp",
        selector_namespace="canary",
        selector_version=17,
        profile="production",
        run_id="run-cancel-1",
    )
    builder = getattr(task_builder, builder_name)

    ok, message = builder(
        clip_list=[
            {"url": "https://example.com/a.mp4"},
            {"url": "https://example.com/b.mp4"},
            {"url": "https://example.com/c.mp4"},
        ],
        run_id="join-run-cancel",
        join_settings={
            "model": "wan_2_2_vace_lightning_baseline_2_2_2",
            "guidance_kind": "vace",
            "continuity_case": "join_bridge",
            "wgp_profile": "default",
            "fps": 16,
        },
        per_join_settings=[],
        vlm_enhanced_prompts=[],
        current_run_output_dir=tmp_path,
        orchestrator_task_id_str="join-orch-cancel",
        orchestrator_project_id="project-1",
        orchestrator_payload={"fps": 16},
        parent_generation_id="parent-1",
        parent_params={"route_contract": parent_route_contract},
        parent_route_key="join_clips_orchestrator",
    )

    assert ok is False
    assert "Orchestrator cancelled" in message
    assert expected_cancel_context in message
    assert [call["task_type_str"] for call in calls] == ["join_clips_segment"]
    assert calls[0]["route_snapshot_fields"]["selector_version"] == 17
    assert calls[0]["route_snapshot_fields"]["route_selection_snapshot"]["parent_route_key"] == "join_clips_orchestrator"
    assert cancelled == [("join-orch-cancel", "Orchestrator cancelled by user")]


def test_join_existing_child_route_mismatch_returns_repair_required(
    routing, monkeypatch
) -> None:
    from source.core.params.task_result import TaskOutcome
    from source.task_handlers.join import shared as join_shared

    parent_route_contract = routing.route_snapshot_fields(
        task_id="edit-parent-1",
        task_type="edit_video_orchestrator",
        params={},
        backend="wgp",
        selector_namespace="canary",
        selector_version=12,
        profile="production",
    )
    child_route_contract = routing.parent_derived_child_route_snapshot_fields(
        parent_params={"route_contract": parent_route_contract},
        child_task_type="join_clips_segment",
        child_params={
            "model": "wan_2_2_vace_lightning_baseline_2_2_2",
            "guidance_kind": "vace",
            "continuity_case": "join_bridge",
            "wgp_profile": "default",
        },
    )
    child_route_contract["selector_version"] = 13

    monkeypatch.setattr(
        join_shared,
        "get_orchestrator_child_tasks",
        lambda _task_id: {
            "join_clips_segment": [
                {
                    "id": "join-child-1",
                    "status": "In Progress",
                    "params": {"route_contract": child_route_contract},
                }
            ],
            "join_final_stitch": [],
        },
    )

    result = join_shared._check_existing_join_tasks(
        "edit-parent-1",
        1,
        parent_params={"route_contract": parent_route_contract},
        expected_parent_route_key="edit_video_orchestrator",
    )

    assert result is not None
    assert result.outcome == TaskOutcome.FAILED
    assert result.error_message
    assert "[ROUTE_REPAIR_REQUIRED]" in result.error_message
    assert "selector_version" in result.error_message


def test_travel_existing_child_route_mismatch_returns_repair_required(
    routing,
) -> None:
    from source.core.params.task_result import TaskOutcome
    from source.task_handlers.travel.orchestration import orchestrator as travel_orchestrator

    parent_route_contract = routing.route_snapshot_fields(
        task_id="travel-parent-1",
        task_type="travel_orchestrator",
        params={},
        backend="wgp",
        selector_namespace="canary",
        selector_version=12,
        profile="production",
    )
    child_route_contract = routing.parent_derived_child_route_snapshot_fields(
        parent_params={"route_contract": parent_route_contract},
        child_task_type="travel_segment",
        child_params={},
    )
    child_route_contract["selected_profile"] = "3"

    result = travel_orchestrator._travel_handle_existing_children_idempotency(
        orchestrator_task_id_str="travel-parent-1",
        parent_params={"route_contract": parent_route_contract},
        existing_segments=[
            {
                "id": "travel-child-1",
                "status": "In Progress",
                "params": {"segment_index": 0, "route_contract": child_route_contract},
            }
        ],
        existing_stitch=[],
        existing_join_orchestrators=[],
        expected_segments=1,
        required_stitch_count=0,
        required_join_orchestrator_count=0,
    )

    assert result is not None
    assert result.outcome == TaskOutcome.FAILED
    assert result.error_message
    assert "[ROUTE_REPAIR_REQUIRED]" in result.error_message
    assert "selected_profile" in result.error_message


def _debug_route_contract(
    routing,
    *,
    task_id: str = "task-1",
    task_type: str = "travel_segment",
    backend: str = "wgp",
    selector_namespace: str = "canary",
    selector_version: int = 12,
    profile: str = "production",
    parent_route_key: str = "travel_orchestrator",
) -> dict:
    contract = routing.route_snapshot_fields(
        task_id=task_id,
        task_type=task_type,
        params={},
        backend=backend,
        selector_namespace=selector_namespace,
        selector_version=selector_version,
        profile=profile,
    )
    contract["route_selection_snapshot"]["parent_route_key"] = parent_route_key
    return contract


def test_debug_route_contract_summary_reads_nested_params(routing) -> None:
    from debug import diagnostics as debug_diagnostics

    task = {
        "id": "child-1",
        "params": {
            "route_contract": _debug_route_contract(
                routing,
                task_type="join_final_stitch",
                parent_route_key="join_clips_orchestrator",
            )
        },
    }

    summary = debug_diagnostics.extract_route_contract_summary(task)
    assert summary == {
        "route_key": "join_final_stitch",
        "selected_backend": "wgp",
        "selector_version": 12,
        "support_state": "wgp_only",
        "selected_profile": "production",
        "parent_route_key": "join_clips_orchestrator",
    }

    formatted = debug_diagnostics.format_route_contract_summary(task)
    assert "route=join_final_stitch" in formatted
    assert "backend=wgp" in formatted
    assert "selector_version=12" in formatted
    assert "support=wgp_only" in formatted
    assert "profile=production" in formatted
    assert "parent=join_clips_orchestrator" in formatted


def test_route_repair_signals_cover_lifecycle_diagnostics(routing) -> None:
    from debug import diagnostics as debug_diagnostics

    parent = {
        "id": "travel-parent-1",
        "task_type": "travel_orchestrator",
        "status": "Failed",
        "error_message": "[ROUTE_REPAIR_REQUIRED] mixed child contracts",
        "params": {
            "route_contract": _debug_route_contract(
                routing,
                task_id="travel-parent-1",
                task_type="travel_orchestrator",
                parent_route_key="",
            )
        },
    }
    child = {
        "id": "travel-child-1",
        "task_type": "travel_segment",
        "status": "In Progress",
        "output_location": "https://storage.example/generated.mp4",
        "params": {
            "segment_index": 0,
            "route_contract": _debug_route_contract(
                routing,
                backend="vibecomfy",
                selector_version=13,
                profile="canary-profile",
                parent_route_key="join_clips_orchestrator",
            ),
        },
    }

    signals = debug_diagnostics.route_repair_signals(
        child,
        parent_task=parent,
        siblings=[child],
    )
    assert "partial_child" in signals
    assert "uploaded_but_not_completed" in signals
    assert "mixed_backend" in signals
    assert "mixed_selector_version" in signals
    assert "mixed_selected_profile" in signals
    assert "wrong_parent_route_key" in signals

    complete_a = {
        "id": "travel-child-complete-1",
        "task_type": "travel_segment",
        "status": "Complete",
        "output_location": "https://storage.example/a.mp4",
        "params": {
            "segment_index": 0,
            "route_contract": _debug_route_contract(routing),
        },
    }
    complete_b = {
        "id": "travel-child-complete-2",
        "task_type": "travel_segment",
        "status": "Complete",
        "output_location": "https://storage.example/b.mp4",
        "params": {
            "segment_index": 0,
            "route_contract": _debug_route_contract(routing),
        },
    }
    duplicate_signals = debug_diagnostics.route_repair_signals(
        complete_a,
        parent_task=parent,
        siblings=[complete_a, complete_b],
    )
    assert "duplicate_completion_candidate" in duplicate_signals

    parent_signals = debug_diagnostics.route_repair_signals(parent)
    assert "parent_repair_required" in parent_signals


def test_task_formatter_displays_route_contract_and_repair_signals(routing) -> None:
    from debug.formatters import Formatter
    from debug.models import TaskInfo

    parent = {
        "id": "travel-parent-1",
        "task_type": "travel_orchestrator",
        "status": "Failed",
        "worker_id": "worker-1",
        "attempts": 1,
        "error_message": "[ROUTE_REPAIR_REQUIRED] mixed child contracts",
        "params": {
            "route_contract": _debug_route_contract(
                routing,
                task_id="travel-parent-1",
                task_type="travel_orchestrator",
                parent_route_key="",
            )
        },
    }
    child = {
        "id": "travel-child-1",
        "task_type": "travel_segment",
        "status": "In Progress",
        "attempts": 0,
        "output_location": "https://storage.example/generated.mp4",
        "params": {
            "segment_index": 0,
            "route_contract": _debug_route_contract(
                routing,
                backend="vibecomfy",
                selector_version=13,
                profile="canary-profile",
                parent_route_key="join_clips_orchestrator",
            ),
        },
    }

    formatted = Formatter.format_task(
        TaskInfo(
            task_id="travel-parent-1",
            state=parent,
            logs=[],
            child_tasks=[child],
        )
    )

    assert "Route: route=travel_orchestrator backend=wgp" in formatted
    assert "Repair Signals: parent_repair_required" in formatted
    assert "Route: route=travel_segment__" in formatted
    assert "backend=vibecomfy" in formatted
    assert "mixed_backend" in formatted
    assert "wrong_parent_route_key" in formatted
