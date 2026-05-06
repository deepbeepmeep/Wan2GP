from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.modules.setdefault("httpx", SimpleNamespace(HTTPError=Exception))
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
        "route_selection_snapshot": {
            "selector_namespace": "production",
            "route_key": route_key,
            "selected_backend": "wgp",
            "selector_version": 42,
            "support_state": "vibecomfy_unsupported",
            "template_id": None,
            "task_id": "child-1",
            "parent_route_key": "join_clips_orchestrator",
        },
    }


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
        "selector_namespace": "staging",
    }


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
        "route_selection_snapshot": {
            "selector_namespace": "production",
            "route_key": "join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default",
            "selected_backend": "wgp",
            "selector_version": 12,
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


def test_parallel_join_child_creation_snapshots_transitions_and_leaves_final_null(monkeypatch, tmp_path) -> None:
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
    )

    assert ok is True
    assert "parallel transitions" in message
    assert [call["task_type_str"] for call in calls] == [
        "join_clips_segment",
        "join_clips_segment",
        "join_final_stitch",
    ]
    for call in calls[:2]:
        assert call["route_snapshot_fields"]["route_key"] == (
            "join_clips_segment__model-wan22_vace__guidance-vace__"
            "continuity-join_bridge__profile-default"
        )
        assert call["route_snapshot_fields"]["selected_backend"] == "wgp"
        assert call["route_snapshot_fields"]["route_selection_snapshot"]["parent_route_key"] == "join_clips_orchestrator"
    assert calls[2]["route_snapshot_fields"] is None
