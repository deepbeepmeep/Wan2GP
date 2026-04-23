"""Live-test task matrix definitions and execution helpers."""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from scripts.create_test_task import TEST_TASKS
from scripts.live_test import config
from scripts.live_test.completion_poller import TaskResult, poll_until_complete
from scripts.live_test.task_spoofer import insert_spoof_task, load_fixture


TRAVEL_WAN_FIXTURE_KEY = "travel_orchestrator_wan2_1seg"
TRAVEL_LTX_FIXTURE_KEY = "travel_orchestrator_ltx"


@dataclass(frozen=True)
class MatrixCase:
    name: str
    task_type: str
    fixture_key: str
    param_overrides: dict[str, Any] = field(default_factory=dict)
    timeout_sec: int = 0


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _anchor_pair() -> list[str]:
    return [config.ANCHOR_IMAGE_A_URL, config.ANCHOR_IMAGE_B_URL]


def _build_wan_travel_fixture() -> dict[str, Any]:
    template = copy.deepcopy(TEST_TASKS["travel_orchestrator"])
    orchestrator_details = template["params"]["orchestrator_details"]
    orchestrator_details["input_image_paths_resolved"] = _anchor_pair()
    orchestrator_details["input_image_generation_ids"] = [
        "live-test-anchor-a",
        "live-test-anchor-b",
    ]
    orchestrator_details["num_new_segments_to_generate"] = 1
    orchestrator_details["segment_frames_expanded"] = [65]
    orchestrator_details["frame_overlap_expanded"] = [10]
    orchestrator_details["base_prompt"] = "A smooth cinematic move bridging two high-contrast scenes"
    orchestrator_details["base_prompts_expanded"] = [orchestrator_details["base_prompt"]]
    orchestrator_details["negative_prompts_expanded"] = [""]
    orchestrator_details["enhanced_prompts_expanded"] = [""]
    return template


def _build_ltx_travel_fixture() -> dict[str, Any]:
    template = _build_wan_travel_fixture()
    orchestrator_details = template["params"]["orchestrator_details"]
    orchestrator_details["model_name"] = config.LTX_MODEL_ID
    orchestrator_details["model_type"] = "ltx2"
    orchestrator_details["parsed_resolution_wh"] = "768x512"
    orchestrator_details["steps"] = 8
    orchestrator_details["fps_helpers"] = 24
    orchestrator_details["guidance_scale"] = 3.0
    orchestrator_details["num_inference_steps"] = 8
    orchestrator_details["flow_shift"] = 5
    orchestrator_details["enhance_prompt"] = False
    orchestrator_details["frame_overlap_expanded"] = [25]
    orchestrator_details["travel_guidance"] = {"kind": "none"}
    orchestrator_details.pop("phase_config", None)
    orchestrator_details.pop("selected_phase_preset_id", None)
    return template


def resolve_case_fixture(case: MatrixCase) -> dict[str, Any]:
    if case.fixture_key == TRAVEL_WAN_FIXTURE_KEY:
        return _build_wan_travel_fixture()
    if case.fixture_key == TRAVEL_LTX_FIXTURE_KEY:
        return _build_ltx_travel_fixture()
    return load_fixture(case.fixture_key)


def build_case_params_overrides(
    case: MatrixCase,
    *,
    unique_suffix: str | None = None,
) -> dict[str, Any]:
    suffix = unique_suffix or uuid.uuid4().hex[:12]
    task_marker = f"live-test-{case.name}-{suffix}"
    runtime: dict[str, Any] = {"task_id": task_marker}

    if case.task_type == "travel_orchestrator":
        runtime["orchestrator_details"] = {
            "run_id": task_marker,
            "orchestrator_task_id": task_marker,
            "input_image_generation_ids": [
                f"{task_marker}-anchor-a",
                f"{task_marker}-anchor-b",
            ],
        }

    if case.task_type == "individual_travel_segment":
        runtime["orchestrator_details"] = {
            "orchestrator_task_id": f"{task_marker}-parent",
            "input_image_paths_resolved": _anchor_pair(),
        }
        runtime["individual_segment_params"] = {
            "start_image_url": config.ANCHOR_IMAGE_A_URL,
            "end_image_url": config.ANCHOR_IMAGE_B_URL,
            "input_image_paths_resolved": _anchor_pair(),
        }
        runtime["start_image_url"] = config.ANCHOR_IMAGE_A_URL
        runtime["end_image_url"] = config.ANCHOR_IMAGE_B_URL
        runtime["input_image_paths_resolved"] = _anchor_pair()

    return _deep_merge(case.param_overrides, runtime)


def render_case_payload(
    case: MatrixCase,
    *,
    project_id: str,
    unique_suffix: str | None = None,
) -> dict[str, Any]:
    payload = copy.deepcopy(resolve_case_fixture(case))
    payload.pop("notes", None)
    payload.pop("description", None)
    payload["project_id"] = project_id
    payload["task_type"] = case.task_type
    payload["status"] = "Queued"

    params = payload.get("params")
    if not isinstance(params, dict):
        params = {}
    payload["params"] = _deep_merge(
        params,
        build_case_params_overrides(case, unique_suffix=unique_suffix),
    )
    payload["params"]["live_test"] = True
    return payload


def build_matrix(
    *,
    anchor_image_a: str = config.ANCHOR_IMAGE_A_URL,
    anchor_image_b: str = config.ANCHOR_IMAGE_B_URL,
    timeout_image_sec: int = config.TIMEOUT_IMAGE_SEC,
    timeout_travel_segment_sec: int = config.TIMEOUT_INDIVIDUAL_TRAVEL_SEGMENT_SEC,
    timeout_travel_orchestrator_sec: int = config.TIMEOUT_TRAVEL_ORCHESTRATOR_SEC,
) -> list[MatrixCase]:
    return [
        MatrixCase(
            name="travel_orchestrator_wan2_1seg",
            task_type="travel_orchestrator",
            fixture_key=TRAVEL_WAN_FIXTURE_KEY,
            timeout_sec=timeout_travel_orchestrator_sec,
        ),
        MatrixCase(
            name="travel_orchestrator_ltx",
            task_type="travel_orchestrator",
            fixture_key=TRAVEL_LTX_FIXTURE_KEY,
            timeout_sec=timeout_travel_orchestrator_sec,
        ),
        MatrixCase(
            name="individual_travel_segment",
            task_type="individual_travel_segment",
            fixture_key="wan22_i2v_individual_segment",
            param_overrides={
                "start_image_url": anchor_image_a,
                "end_image_url": anchor_image_b,
                "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
                "orchestrator_details": {
                    "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
                },
                "individual_segment_params": {
                    "start_image_url": anchor_image_a,
                    "end_image_url": anchor_image_b,
                    "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
                },
            },
            timeout_sec=timeout_travel_segment_sec,
        ),
        MatrixCase(
            name="qwen_image_style",
            task_type="qwen_image_style",
            fixture_key="qwen_image_style_db_task",
            param_overrides={
                "style_reference_image": anchor_image_a,
                "subject_reference_image": anchor_image_a,
            },
            timeout_sec=timeout_image_sec,
        ),
        MatrixCase(
            name="qwen_image_t2i",
            task_type="qwen_image",
            fixture_key="qwen_image_basic",
            timeout_sec=timeout_image_sec,
        ),
        MatrixCase(
            name="qwen_image_2512",
            task_type="qwen_image_2512",
            fixture_key="qwen_image_basic",
            param_overrides={"resolution": "1536x864"},
            timeout_sec=timeout_image_sec,
        ),
        MatrixCase(
            name="qwen_image_edit",
            task_type="qwen_image_edit",
            fixture_key="qwen_image_edit_basic",
            param_overrides={
                "image": anchor_image_b,
                "image_url": anchor_image_b,
            },
            timeout_sec=timeout_image_sec,
        ),
        MatrixCase(
            name="z_image_turbo_i2i",
            task_type="z_image_turbo_i2i",
            fixture_key="z_image_turbo_i2i_basic",
            param_overrides={"image_url": anchor_image_a},
            timeout_sec=timeout_image_sec,
        ),
    ]


MATRIX = build_matrix()


def run_matrix(db, project_id: str, cases: list[MatrixCase]) -> list[TaskResult]:
    results: list[TaskResult] = []
    for case in cases:
        started = time.monotonic()
        try:
            suffix = uuid.uuid4().hex[:12]
            fixture_payload = resolve_case_fixture(case)
            overrides = build_case_params_overrides(case, unique_suffix=suffix)
            task_id = insert_spoof_task(
                db,
                project_id,
                case.task_type,
                overrides,
                fixture_payload=fixture_payload,
            )
            result = poll_until_complete(
                db,
                task_id,
                project_id,
                timeout_sec=case.timeout_sec,
                case_name=case.name,
                task_type=case.task_type,
            )
        except Exception as exc:
            result = TaskResult(
                task_id=f"insert-failed:{case.name}",
                case_name=case.name,
                task_type=case.task_type,
                final_status="Insert Failed",
                output_location=None,
                generation_ids=[],
                elapsed_sec=round(time.monotonic() - started, 3),
                error_summary=str(exc),
            )
        results.append(result)
    return results


__all__ = [
    "MATRIX",
    "MatrixCase",
    "TRAVEL_LTX_FIXTURE_KEY",
    "TRAVEL_WAN_FIXTURE_KEY",
    "build_case_params_overrides",
    "build_matrix",
    "render_case_payload",
    "resolve_case_fixture",
    "run_matrix",
]
