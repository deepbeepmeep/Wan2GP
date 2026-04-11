from __future__ import annotations

import copy
from pathlib import Path

DEFAULT_FIXTURE_ORDER = [
    "travel_orchestrator",
    "join_clips_orchestrator",
    "individual_travel_segment",
    "qwen_image",
    "z_image_turbo",
]

FIXTURE_IDS = {
    "travel_orchestrator": "preview-travel-orch-0001",
    "join_clips_orchestrator": "preview-join-orch-0002",
    "individual_travel_segment": "preview-segment-0003",
    "qwen_image": "preview-qwen-0004",
    "z_image_turbo": "preview-zimage-0005",
}


def _normalize_filter_types(filter_types):
    if filter_types is None:
        return set(DEFAULT_FIXTURE_ORDER)
    if isinstance(filter_types, str):
        values = [item.strip() for item in filter_types.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in filter_types if str(item).strip()]
    normalized = {"z_image_turbo" if value == "z_image" else value for value in values}
    return set(normalized)


def _sanitize_value(value, asset_paths: dict[str, str]):
    if isinstance(value, dict):
        return {key: _sanitize_value(item, asset_paths) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item, asset_paths) for item in value]
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        lowered = value.lower()
        if lowered.endswith(".safetensors"):
            return "file:///preview/fake_lora.safetensors"
        if lowered.endswith((".mp4", ".mov", ".webm")):
            return asset_paths["guide_video"] if "guide" in lowered or "structure" in lowered else asset_paths["video"]
        return asset_paths["image"]
    return value


def _load_test_tasks():
    from scripts.create_test_task import TEST_TASKS

    return TEST_TASKS


def _build_travel_orchestrator(asset_paths: dict[str, str]) -> dict:
    template = copy.deepcopy(_load_test_tasks()["travel_orchestrator"])
    params = _sanitize_value(template["params"], asset_paths)
    details = params["orchestrator_details"]
    details["run_id"] = "preview-travel-run-0001"
    details["orchestrator_task_id"] = FIXTURE_IDS["travel_orchestrator"]
    details["main_output_dir_for_run"] = "./outputs/preview_travel_output"
    details["input_image_generation_ids"] = ["preview-image-0001"]
    return {
        "task_id": FIXTURE_IDS["travel_orchestrator"],
        "task_type": "travel_orchestrator",
        "params": params,
        "project_id": template["project_id"],
        "attempts": 0,
    }


def _build_join_clips_orchestrator(asset_paths: dict[str, str]) -> dict:
    phase_config = copy.deepcopy(_load_test_tasks()["travel_orchestrator"]["params"]["orchestrator_details"]["phase_config"])
    params = {
        "orchestrator_details": {
            "run_id": "preview-join-run-0002",
            "clip_list": [
                {"url": asset_paths["video"], "name": "clip_a"},
                {"url": asset_paths["video"], "name": "clip_b"},
                {"url": asset_paths["video"], "name": "clip_c"},
            ],
            "loop_first_clip": False,
            "context_frame_count": 8,
            "gap_frame_count": 19,
            "replace_mode": True,
            "prompt": "Preview join transition",
            "negative_prompt": "",
            "model": "wan_2_2_vace_lightning_baseline_2_2_2",
            "phase_config": _sanitize_value(phase_config, asset_paths),
            "additional_loras": {},
            "seed": 20260410,
            "resolution": "902x508",
            "fps": 16,
            "use_input_video_resolution": True,
            "use_input_video_fps": True,
            "output_base_dir": "./outputs/preview_join_output",
            "use_parallel_joins": True,
        }
    }
    return {
        "task_id": FIXTURE_IDS["join_clips_orchestrator"],
        "task_type": "join_clips_orchestrator",
        "params": params,
        "project_id": _load_test_tasks()["travel_orchestrator"]["project_id"],
        "attempts": 0,
    }


def _build_individual_travel_segment(asset_paths: dict[str, str]) -> dict:
    template = copy.deepcopy(_load_test_tasks()["uni3c_basic"])
    params = _sanitize_value(template["params"], asset_paths)
    params["orchestrator_details"]["orchestrator_task_id"] = "preview-segment-parent-0003"
    return {
        "task_id": FIXTURE_IDS["individual_travel_segment"],
        "task_type": "individual_travel_segment",
        "params": params,
        "project_id": template["project_id"],
        "attempts": 0,
    }


def _build_qwen_image(asset_paths: dict[str, str]) -> dict:
    template = copy.deepcopy(_load_test_tasks()["qwen_image_style"])
    params = _sanitize_value(template["params"], asset_paths)
    params["task_id"] = FIXTURE_IDS["qwen_image"]
    params["model"] = "qwen_image_20B"
    params["prompt"] = "Preview qwen image"
    params["image_url"] = asset_paths["image"]
    params["resolution"] = "1024x1024"
    params["num_inference_steps"] = 4
    return {
        "task_id": FIXTURE_IDS["qwen_image"],
        "task_type": "qwen_image",
        "params": params,
        "project_id": template["project_id"],
        "attempts": 0,
    }


def _build_z_image_turbo(asset_paths: dict[str, str]) -> dict:
    template = copy.deepcopy(_load_test_tasks()["qwen_image_style"])
    params = _sanitize_value(template["params"], asset_paths)
    params["task_id"] = FIXTURE_IDS["z_image_turbo"]
    params["model"] = "z_image"
    params["prompt"] = "Preview z image"
    params["resolution"] = "512x512"
    params["num_inference_steps"] = 8
    params.pop("style_reference_image", None)
    params.pop("subject_reference_image", None)
    return {
        "task_id": FIXTURE_IDS["z_image_turbo"],
        "task_type": "z_image_turbo",
        "params": params,
        "project_id": template["project_id"],
        "attempts": 0,
    }


def get_fixtures(filter_types=None) -> list[dict]:
    from scripts.preview.assets import ensure_assets

    selected = _normalize_filter_types(filter_types)
    asset_paths = ensure_assets(Path(__file__).resolve().parent)
    builders = {
        "travel_orchestrator": _build_travel_orchestrator,
        "join_clips_orchestrator": _build_join_clips_orchestrator,
        "individual_travel_segment": _build_individual_travel_segment,
        "qwen_image": _build_qwen_image,
        "z_image_turbo": _build_z_image_turbo,
    }
    return [builders[name](asset_paths) for name in DEFAULT_FIXTURE_ORDER if name in selected]
