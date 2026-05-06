from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from scripts.dual_run_compare.thresholds import DEFAULT_PATH


DUAL_RUN_DIR = DEFAULT_PATH.parent
NON_RAYWORKER_DIR = DUAL_RUN_DIR / "fixtures" / "non_rayworker"
ORCHESTRATOR_REGISTRY = (
    Path(__file__).parents[4]
    / "reigh-worker-orchestrator"
    / "api_orchestrator"
    / "task_handlers.py"
)

REQUIRED_ROUTE_KEYS = {
    "video_enhance",
    "image-upscale",
    "animate_character",
    "flux_klein_edit",
}

REQUIRED_FIXTURE_FIELDS = {
    "fixture_schema_version",
    "route_key",
    "task_type",
    "runtime",
    "payload_shape",
    "completion_handler",
    "billing_path",
    "output_shape",
    "owner",
    "preserve_or_move_decision",
    "alias_mismatch",
    "executor_handler",
    "external_api",
    "canary_depth",
    "landed_status",
    "oracle_policy",
    "report_status_policy",
    "required_full_canary_sections",
    "route_classification",
    "shared_path_policy",
}

REQUIRED_FULL_CANARY_SECTIONS = {
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

REQUIRED_SHARED_PATH_KEYS = {
    "billing",
    "completion",
    "queue_contract",
    "shadow_side_effects",
}

EXPECTED_HANDLERS = {
    "video_enhance": "handle_video_enhance",
    "image-upscale": "handle_image_upscale",
    "animate_character": "handle_animate_character",
    "flux_klein_edit": "handle_flux_klein_edit",
}

EXPECTED_PAYLOAD_KEYS = {
    "video_enhance": {"video_url", "video", "enable_interpolation", "enable_upscale", "interpolation", "upscale"},
    "image-upscale": {"image_url", "image", "upscale_factor", "scale_factor", "noise_scale", "output_format", "seed"},
    "animate_character": {"character_image_url", "motion_video_url", "mode", "prompt", "resolution", "seed"},
    "flux_klein_edit": {
        "image_url",
        "image",
        "prompt",
        "klein_model",
        "image_size",
        "target_megapixels",
        "num_inference_steps",
        "num_images",
    },
}


def _load_fixture(route_key: str) -> dict:
    return json.loads((NON_RAYWORKER_DIR / f"{route_key}.json").read_text())


def test_non_rayworker_fixture_json_and_snapshot_are_complete() -> None:
    fixtures = {path.stem: json.loads(path.read_text()) for path in NON_RAYWORKER_DIR.glob("*.json")}
    snapshot = fixtures.pop("registry_snapshot")

    assert set(fixtures) == REQUIRED_ROUTE_KEYS
    assert REQUIRED_ROUTE_KEYS <= set(snapshot["routes"])
    assert snapshot["snapshot_schema_version"] == 2
    assert snapshot["active_api_owned_route_count"] == 14

    for route_key, fixture in fixtures.items():
        assert REQUIRED_FIXTURE_FIELDS <= set(fixture)
        assert fixture["fixture_schema_version"] == 1
        assert fixture["route_key"] == route_key
        assert fixture["task_type"] == route_key
        assert fixture["runtime"] == "api_orchestrator"
        assert fixture["preserve_or_move_decision"] == "preserve"
        assert fixture["route_classification"] == "active_api_owned"
        assert fixture["canary_depth"] == "full_canary"
        assert fixture["landed_status"] == "landed_full_canary"
        assert fixture["report_status_policy"] == "red_or_green_required"
        assert fixture["oracle_policy"] == "full_product_billing_completion"
        assert set(fixture["required_full_canary_sections"]) == REQUIRED_FULL_CANARY_SECTIONS
        assert set(fixture["shared_path_policy"]) == REQUIRED_SHARED_PATH_KEYS
        assert set(fixture["shared_path_policy"].values()) == {"required_full"}
        assert fixture["completion_handler"] == "complete_task/generation-handlers.ts"
        assert fixture["billing_path"] == "complete_task/billing.ts"
        assert isinstance(fixture["payload_shape"], dict) and fixture["payload_shape"]
        assert isinstance(fixture["output_shape"], dict) and fixture["output_shape"]
        assert set(fixture["payload_shape"]) == EXPECTED_PAYLOAD_KEYS[route_key]

        snapshot_route = snapshot["routes"][route_key]
        assert snapshot_route["task_type"] == route_key
        assert snapshot_route["runtime"] == fixture["runtime"]
        assert snapshot_route["executor_handler"] == fixture["executor_handler"]
        assert snapshot_route["canary_depth"] == "full_canary"
        assert snapshot_route["landed_status"] == "landed_full_canary"
        assert snapshot_route["report_status_policy"] == "red_or_green_required"
        assert snapshot_route["oracle_policy"] == "full_product_billing_completion"
        assert snapshot_route["shared_path_policy"] == fixture["shared_path_policy"]


def test_image_upscale_records_alias_mismatch() -> None:
    fixture = _load_fixture("image-upscale")
    assert fixture["alias_mismatch"] == {
        "legacy_db_row": "image_upscale",
        "resolver_and_api_handler": "image-upscale",
    }

    for route_key in REQUIRED_ROUTE_KEYS - {"image-upscale"}:
        assert _load_fixture(route_key)["alias_mismatch"] is None


def test_live_orchestrator_registry_matches_snapshot_when_sibling_repo_exists() -> None:
    if not ORCHESTRATOR_REGISTRY.is_file():
        pytest.skip("sibling reigh-worker-orchestrator checkout is absent")

    tree = ast.parse(ORCHESTRATOR_REGISTRY.read_text())
    task_handlers: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "TASK_HANDLERS" for target in node.targets):
            continue
        assert isinstance(node.value, ast.Dict)
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(value, ast.Name):
                task_handlers[key.value] = value.id

    snapshot = json.loads((NON_RAYWORKER_DIR / "registry_snapshot.json").read_text())
    for route_key, expected_handler in EXPECTED_HANDLERS.items():
        assert task_handlers[route_key] == expected_handler
        assert snapshot["routes"][route_key]["executor_handler"].endswith(f"::{expected_handler}")


def test_all_active_api_owned_task_handlers_have_policy_classification() -> None:
    if not ORCHESTRATOR_REGISTRY.is_file():
        pytest.skip("sibling reigh-worker-orchestrator checkout is absent")

    tree = ast.parse(ORCHESTRATOR_REGISTRY.read_text())
    task_handlers: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "TASK_HANDLERS" for target in node.targets):
            continue
        assert isinstance(node.value, ast.Dict)
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(value, ast.Name):
                task_handlers[key.value] = value.id

    snapshot = json.loads((NON_RAYWORKER_DIR / "registry_snapshot.json").read_text())
    snapshot_routes = snapshot["routes"]
    api_owned_routes = {
        route_key
        for route_key in task_handlers
        if not route_key.startswith("banodoco_")
    }

    assert api_owned_routes <= set(snapshot_routes)
    assert snapshot["active_api_owned_route_count"] == len(api_owned_routes)
    for route_key in api_owned_routes:
        route = snapshot_routes[route_key]
        assert route["route_classification"] == "active_api_owned"
        assert route["runtime"] == "api_orchestrator"
        assert route["executor_handler"].endswith(f"::{task_handlers[route_key]}")
        assert set(route["shared_path_policy"]) == REQUIRED_SHARED_PATH_KEYS
        assert route["report_status_policy"] in {
            "red_or_green_required",
            "pending_until_shared_oracle_evidence",
            "wgp_only",
        }
        assert route["oracle_policy"] in {
            "full_product_billing_completion",
            "lightweight_shared_path",
        }

    assert snapshot_routes["z_image_turbo"]["landed_status"] == "sprint2_vibecomfy_direct_default_resolution_landed"
    assert snapshot_routes["z_image_turbo"]["report_status_policy"] == "red_or_green_required"
    for route_key in {
        "qwen_image",
        "qwen_image_2512",
        "qwen_image_edit",
        "qwen_image_style",
        "image_inpaint",
        "annotated_image_edit",
    }:
        assert snapshot_routes[route_key]["landed_status"] == "sprint5_wgp_only"
        assert snapshot_routes[route_key]["report_status_policy"] == "wgp_only"
    assert snapshot_routes["wan_2_2_t2i"]["landed_status"] == "sprint2_wgp_only"
    assert snapshot_routes["wan_2_2_t2i"]["report_status_policy"] == "wgp_only"

    for route_key in {"banodoco_timeline_generate", "banodoco_render_timeline"}:
        route = snapshot_routes[route_key]
        assert route["route_classification"] == "worker_pool_fallback"
        assert route["report_status_policy"] == "fallback"
