from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from source.media.video.vace_frame_utils import prepare_vace_generation_params
from source.task_handlers.contracts.dispatch import normalize_task_dispatch_payload
from source.task_handlers.queue import download_ops
from source.task_handlers.queue.task_queue import GenerationTask
from source.task_handlers.tasks import task_registry
from source.task_handlers.tasks.task_registry import _inject_ic_lora_with_dedup
from source.task_handlers.tasks.template_routing import (
    RouteSupportState,
    WorkerBackend,
    resolve_task_route,
)


ROUTE_KEYS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dual_run_compare" / "route_keys.py"
WAN_VACE_MODEL = "wan_2_2_vace_lightning_baseline_2_2_2"
WAN_VACE_PHASE_CONFIG = {
    "num_phases": 3,
    "steps_per_phase": [2, 2, 2],
    "flow_shift": 5,
    "sample_solver": "euler",
    "model_switch_phase": 2,
    "phases": [
        {"guidance_scale": 3},
        {"guidance_scale": 1},
        {"guidance_scale": 1},
    ],
}


def _cohort_e_route_key(**kwargs):
    spec = importlib.util.spec_from_file_location("dual_run_route_keys_under_test", ROUTE_KEYS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        return module.cohort_e_route_key(**kwargs)
    finally:
        sys.modules.pop(spec.name, None)


def test_dedup_updates_strength_when_basename_matches():
    segment_loras = [
        {
            "path": "https://example.com/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
            "strength": 0.4,
        }
    ]
    ic_entry = {
        "path": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "strength": 0.8,
        "name": "ic-lora-pose (auto-injected)",
    }

    result = _inject_ic_lora_with_dedup(segment_loras, ic_entry)

    assert len(result) == 1
    assert result[0]["strength"] == 0.8


def test_dedup_appends_when_no_match():
    ic_entry = {
        "path": "https://example.com/ic-lora-cameraman.safetensors",
        "strength": 0.7,
        "name": "ic-lora-cameraman (auto-injected)",
    }

    result = _inject_ic_lora_with_dedup([], ic_entry)

    assert len(result) == 1
    assert result[0] == ic_entry


def test_wan_vace_travel_child_smoke_uses_constructed_payload_defaults_and_dependency(tmp_path):
    payload = {
        "task_type": "travel_segment",
        "dependant_on": "previous-segment-task",
        "model_name": WAN_VACE_MODEL,
        "model_family": "wan22_vace",
        "travel_guidance": {
            "kind": "vace",
            "mode": "raw",
            "videos": [{"path": str(tmp_path / "guide.mp4"), "start_frame": 0, "end_frame": 48}],
        },
        "continuity_case": "video_source",
        "video_source": str(tmp_path / "prefix.mp4"),
        "override_profile": "default",
        "orchestrator_details": {"run_id": "run-1", "model_name": WAN_VACE_MODEL},
    }

    normalized = normalize_task_dispatch_payload(payload, task_id="travel-child-1")
    resolved = resolve_task_route(
        task_id=normalized["task_id"],
        task_type="travel_segment",
        params=normalized,
        backend=WorkerBackend.VIBECOMFY,
    )

    assert normalized["dependant_on"] == "previous-segment-task"
    assert normalized["orchestrator_details"]["orchestrator_task_id"] == "travel-child-1"
    assert (
        resolved.route_key
        == "travel_segment__model-wan22_vace__guidance-vace__continuity-video_source__profile-default"
    )
    assert resolved.support_state == RouteSupportState.VIBECOMFY_UNSUPPORTED
    assert resolved.should_use_vibecomfy is False

    ctx = task_registry.SegmentContext(
        mode="orchestrator",
        orchestrator_details={
            "model_name": WAN_VACE_MODEL,
            "phase_config": WAN_VACE_PHASE_CONFIG,
        },
        individual_params={},
        segment_idx=1,
        segment_params={
            "phase_config": WAN_VACE_PHASE_CONFIG,
            "travel_guidance": payload["travel_guidance"],
        },
        orchestrator_task_id_ref="parent-travel",
        orchestrator_run_id="run-1",
    )
    gen = task_registry.GenerationInputs(
        model_name=WAN_VACE_MODEL,
        prompt_for_wgp="cinematic bridge between anchors",
        negative_prompt_for_wgp="blurry",
        parsed_res_wh=(832, 480),
        total_frames_for_segment=49,
        current_run_base_output_dir=tmp_path,
        segment_processing_dir=tmp_path / "segment",
        debug_enabled=False,
        travel_mode="wan",
        generation_policy=task_registry.GenerationPolicy.from_payload({"model_type": "vace"}),
        model_family="wan",
    )
    image_refs = task_registry.ImageRefs(
        start_ref_path=str(tmp_path / "start.png"),
        end_ref_path=str(tmp_path / "end.png"),
    )
    structure = task_registry.StructureOutputs()

    params = task_registry._build_generation_params(ctx, gen, image_refs, structure, "travel-child-1")

    assert params["model_name"] == WAN_VACE_MODEL
    assert params["video_length"] == 49
    assert params["resolution"] == "832x480"
    assert params["num_inference_steps"] == 6
    assert params["guidance_phases"] == 3
    assert params["guidance_scale"] == 3
    assert params["guidance2_scale"] == 1
    assert params["guidance3_scale"] == 1
    assert params["flow_shift"] == 5
    assert params["sample_solver"] == "euler"
    assert params["model_switch_phase"] == 2


def test_wan_vace_join_child_smoke_uses_constructed_payload_defaults_and_dependency(tmp_path):
    payload = {
        "task_type": "join_clips_segment",
        "dependant_on": ["clip-a-task", "clip-b-task"],
        "model": WAN_VACE_MODEL,
        "model_family": "wan22_vace",
        "guidance_kind": "vace",
        "continuity_case": "join_bridge",
        "wgp_profile": "default",
        "prompt": "smooth temporal transition",
        "negative_prompt": "warped edges",
        "seed": 12345,
        "num_inference_steps": 6,
        "guidance_scale": 3,
        "guidance2_scale": 1,
        "guidance3_scale": 1,
        "guidance_phases": 3,
        "flow_shift": 5,
        "switch_threshold": 883,
        "switch_threshold2": 558,
        "model_switch_phase": 2,
        "sample_solver": "euler",
    }

    normalized = normalize_task_dispatch_payload(payload, task_id="join-child-1")
    route_key = _cohort_e_route_key(
        task_type=normalized["task_type"],
        model_name=normalized["model"],
        model_family=normalized["model_family"],
        guidance_kind=normalized["guidance_kind"],
        continuity_case=normalized["continuity_case"],
        profile=normalized["wgp_profile"],
    )
    params = prepare_vace_generation_params(
        guide_video_path=tmp_path / "join-guide.mp4",
        mask_video_path=tmp_path / "join-mask.mp4",
        total_frames=49,
        resolution_wh=(832, 480),
        prompt=normalized["prompt"],
        negative_prompt=normalized["negative_prompt"],
        model=normalized["model"],
        seed=normalized["seed"],
        task_params=normalized,
    )

    assert normalized["dependant_on"] == ["clip-a-task", "clip-b-task"]
    assert route_key == "join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default"
    assert params["video_prompt_type"] == "VM"
    assert params["video_length"] == 49
    assert params["resolution"] == "832x480"
    assert params["num_inference_steps"] == 6
    assert params["guidance_phases"] == 3
    assert params["guidance_scale"] == 3
    assert params["guidance2_scale"] == 1
    assert params["guidance3_scale"] == 1
    assert params["flow_shift"] == 5
    assert params["switch_threshold"] == 883
    assert params["switch_threshold2"] == 558
    assert params["model_switch_phase"] == 2
    assert params["sample_solver"] == "euler"


def test_wan_vace_individual_travel_segment_uses_real_local_contract_path(tmp_path):
    payload = {
        "task_type": "individual_travel_segment",
        "parent_generation_id": "parent-generation-1",
        "child_generation_id": "child-generation-1",
        "segment_index": 2,
        "start_image_url": str(tmp_path / "start.png"),
        "end_image_url": str(tmp_path / "end.png"),
        "model_name": WAN_VACE_MODEL,
        "model_type": "vace",
        "travel_guidance": {
            "kind": "vace",
            "mode": "raw",
            "videos": [{"path": str(tmp_path / "guide.mp4"), "start_frame": 0, "end_frame": 48}],
        },
        "pair_shot_generation_id": "pair-shot-generation-1",
        "phase_config": WAN_VACE_PHASE_CONFIG,
        "selected_phase_preset_id": "__builtin_vace_default__",
        "num_frames": 49,
        "resolution": "832x480",
    }

    normalized = normalize_task_dispatch_payload(payload, task_id="individual-child-1")
    resolved = resolve_task_route(
        task_id=normalized["task_id"],
        task_type="individual_travel_segment",
        params=normalized,
        backend=WorkerBackend.VIBECOMFY,
    )

    assert resolved.route_key == (
        "individual_travel_segment__model-wan22_vace__guidance-vace__"
        "continuity-first_last__profile-default"
    )
    assert resolved.support_state == RouteSupportState.VIBECOMFY_UNSUPPORTED
    assert normalized["pair_shot_generation_id"] == "pair-shot-generation-1"
    assert normalized["travel_guidance"]["kind"] == "vace"

    ctx = task_registry.SegmentContext(
        mode="individual",
        orchestrator_details={
            "model_name": WAN_VACE_MODEL,
            "phase_config": WAN_VACE_PHASE_CONFIG,
        },
        individual_params=normalized,
        segment_idx=2,
        segment_params={"phase_config": WAN_VACE_PHASE_CONFIG},
        orchestrator_task_id_ref="parent-generation-1",
        orchestrator_run_id="run-individual-1",
    )
    gen = task_registry.GenerationInputs(
        model_name=WAN_VACE_MODEL,
        prompt_for_wgp="single segment contract",
        negative_prompt_for_wgp="blur",
        parsed_res_wh=(832, 480),
        total_frames_for_segment=49,
        current_run_base_output_dir=tmp_path,
        segment_processing_dir=tmp_path / "segment",
        debug_enabled=False,
        travel_mode="wan",
        generation_policy=task_registry.GenerationPolicy.from_payload({"model_type": "vace"}),
        model_family="wan",
    )
    image_refs = task_registry.ImageRefs(
        start_ref_path=str(tmp_path / "start.png"),
        end_ref_path=str(tmp_path / "end.png"),
    )
    structure = task_registry.StructureOutputs()

    params = task_registry._build_generation_params(ctx, gen, image_refs, structure, "individual-child-1")

    assert params["model_name"] == WAN_VACE_MODEL
    assert params["video_length"] == 49
    assert params["guidance_phases"] == 3
    assert params["guidance_scale"] == 3
    assert params["guidance2_scale"] == 1
    assert params["guidance3_scale"] == 1
    assert params["flow_shift"] == 5


def test_direct_wan_2_2_t2i_overwrites_conflicting_video_length(monkeypatch, tmp_path):
    captured = {}

    class Queue:
        def submit_task(self, task):
            captured["task"] = task
            return task.id

        def get_task_status(self, task_id):
            return SimpleNamespace(status="completed", result_path=f"/tmp/{task_id}.mp4")

    monkeypatch.setattr(task_registry.time, "sleep", lambda _seconds: None)
    context = {
        "task_id": "wan-image-1",
        "task_params_dict": {
            "prompt": "a single image",
            "video_length": 81,
            "resolution": "1024x1024",
        },
        "wan2gp_path": str(tmp_path),
        "debug_mode": False,
        "colour_match_videos": False,
        "mask_active_frames": False,
        "task_queue": Queue(),
        "main_output_dir_base": tmp_path,
    }

    ok, result = task_registry.TaskRegistry._handle_direct_queue_task("wan_2_2_t2i", context)

    assert ok is True
    assert result == "/tmp/wan-image-1.mp4"
    assert captured["task"].parameters["video_length"] == 1


def test_wan_2_2_t2i_wgp_conversion_preserves_png_identity_and_single_frame(monkeypatch):
    captured = {}

    class FakeConfig:
        def __init__(self, params):
            self.params = dict(params)
            self.generation = SimpleNamespace(prompt="")
            self.model = ""
            self.lora = SimpleNamespace(has_pending_downloads=lambda: False)

        @classmethod
        def from_db_task(cls, params, **context):
            captured["from_db_task"] = {"params": dict(params), "context": context}
            return cls(params)

        def validate(self):
            return []

        def to_wgp_format(self):
            return dict(self.params)

    monkeypatch.setattr(download_ops, "is_debug_enabled", lambda: False)
    monkeypatch.setattr("source.core.params.TaskConfig", FakeConfig)

    queue = SimpleNamespace(
        logger=SimpleNamespace(
            debug_block=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
            debug_anomaly=lambda *_args, **_kwargs: None,
        ),
        wan_dir="/tmp",
    )
    task = GenerationTask(
        id="wan-db-1",
        model="t2v_2_2",
        prompt="single image from db",
        parameters={
            "_source_task_type": "wan_2_2_t2i",
            "video_length": 81,
            "resolution": "1024x1024",
            "seed": 123,
        },
    )

    wgp_params = download_ops.convert_to_wgp_task_impl(queue, task)

    assert captured["from_db_task"]["params"]["video_length"] == 1
    assert captured["from_db_task"]["context"]["task_type"] == "wan_2_2_t2i"
    assert task.parameters["_source_task_type"] == "wan_2_2_t2i"
    assert task.parameters["video_length"] == 1
    assert wgp_params["_source_task_type"] == "wan_2_2_t2i"
    assert wgp_params["video_length"] == 1
    assert wgp_params["prompt"] == "single image from db"
