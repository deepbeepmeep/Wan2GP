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
    assert set(snapshot["routes"]) == REQUIRED_ROUTE_KEYS
    assert snapshot["snapshot_schema_version"] == 1

    for route_key, fixture in fixtures.items():
        assert REQUIRED_FIXTURE_FIELDS <= set(fixture)
        assert fixture["fixture_schema_version"] == 1
        assert fixture["route_key"] == route_key
        assert fixture["task_type"] == route_key
        assert fixture["runtime"] == "api_orchestrator"
        assert fixture["preserve_or_move_decision"] == "preserve"
        assert fixture["completion_handler"] == "complete_task/generation-handlers.ts"
        assert fixture["billing_path"] == "complete_task/billing.ts"
        assert isinstance(fixture["payload_shape"], dict) and fixture["payload_shape"]
        assert isinstance(fixture["output_shape"], dict) and fixture["output_shape"]
        assert set(fixture["payload_shape"]) == EXPECTED_PAYLOAD_KEYS[route_key]

        snapshot_route = snapshot["routes"][route_key]
        assert snapshot_route["task_type"] == route_key
        assert snapshot_route["runtime"] == fixture["runtime"]
        assert snapshot_route["executor_handler"] == fixture["executor_handler"]


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
