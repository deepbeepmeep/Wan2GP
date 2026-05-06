"""Coverage-focused tests for join/task-registry and related handlers."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
from PIL import Image

import source.media.video.travel_guide as travel_guide
import source.models.model_handlers.qwen_handler as qwen_handler
import source.task_handlers.edit_video_orchestrator as edit_video_orchestrator
import source.task_handlers.join.final_stitch as join_final_stitch
import source.task_handlers.join.orchestrator as join_orchestrator
import source.task_handlers.join.task_builder as join_task_builder
import source.task_handlers.join.vlm_enhancement as join_vlm_enhancement
import source.task_handlers.magic_edit as magic_edit
import source.task_handlers.tasks.task_conversion as task_conversion
import source.task_handlers.tasks.task_registry as task_registry


def test_task_conversion_requires_prompt_for_non_img2img_task():
    with pytest.raises(ValueError):
        task_conversion.db_task_to_generation_task(
            db_task_params={},
            task_id="task-1",
            task_type="travel_segment",
            wan2gp_path=".",
        )


def test_task_registry_get_param_prefers_truthy_but_keeps_explicit_booleans():
    assert task_registry._get_param("k", {"k": ""}, {"k": "fallback"}, prefer_truthy=True) == "fallback"
    assert task_registry._get_param("flag", {"flag": False}, {"flag": True}, prefer_truthy=True) is False


def test_join_task_builder_chain_requires_two_clips(tmp_path):
    ok, message = join_task_builder._create_join_chain_tasks(
        clip_list=[{"url": "a.mp4"}],
        run_id="run-1",
        join_settings={},
        per_join_settings=[],
        vlm_enhanced_prompts=[],
        current_run_output_dir=tmp_path,
        orchestrator_task_id_str="orch-1",
        orchestrator_project_id=None,
        orchestrator_payload={},
        parent_generation_id=None,
    )
    assert ok is False
    assert "at least 2 clips" in message


def test_join_task_builder_parallel_requires_two_clips(tmp_path):
    ok, message = join_task_builder._create_parallel_join_tasks(
        clip_list=[{"url": "a.mp4"}],
        run_id="run-2",
        join_settings={},
        per_join_settings=[],
        vlm_enhanced_prompts=[],
        current_run_output_dir=tmp_path,
        orchestrator_task_id_str="orch-2",
        orchestrator_project_id=None,
        orchestrator_payload={},
        parent_generation_id=None,
    )
    assert ok is False
    assert "at least 2 clips" in message


def test_join_orchestrator_requires_orchestrator_details(tmp_path):
    ok, message = join_orchestrator._handle_join_clips_orchestrator_task(
        task_params_from_db={},
        main_output_dir_base=tmp_path,
        orchestrator_task_id_str="orch-3",
        orchestrator_project_id=None,
    )
    assert ok is False
    assert "orchestrator_details missing" in message


def test_join_orchestrator_requires_run_id(tmp_path):
    ok, message = join_orchestrator._handle_join_clips_orchestrator_task(
        task_params_from_db={
            "orchestrator_details": {"clip_list": [{"url": "a.mp4"}, {"url": "b.mp4"}]}
        },
        main_output_dir_base=tmp_path,
        orchestrator_task_id_str="orch-4",
        orchestrator_project_id=None,
    )
    assert ok is False
    assert "run_id is required" in message


def test_join_final_stitch_chain_mode_requires_transition_ids(tmp_path):
    ok, message = join_final_stitch.handle_join_final_stitch(
        task_params_from_db={"chain_mode": True, "transition_task_ids": []},
        main_output_dir_base=tmp_path,
        task_id="stitch-1",
    )
    assert ok is False
    assert "chain_mode=True" in message


def test_join_final_stitch_requires_at_least_two_clips(tmp_path):
    ok, message = join_final_stitch.handle_join_final_stitch(
        task_params_from_db={"clip_list": [{"url": "a.mp4"}], "transition_task_ids": []},
        main_output_dir_base=tmp_path,
        task_id="stitch-2",
    )
    assert ok is False
    assert "at least 2 clips" in message


def test_join_vlm_prompt_generation_with_fake_extender(tmp_path):
    paths = []
    for i in range(4):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (24, 24), (i * 40, 20, 30)).save(p)
        paths.append(str(p))

    extender = types.SimpleNamespace(
        extend_with_img=lambda **_kwargs: types.SimpleNamespace(prompt="smooth transition prompt")
    )
    prompt = join_vlm_enhancement._generate_join_transition_prompt(
        start_first_path=paths[0],
        start_boundary_path=paths[1],
        end_boundary_path=paths[2],
        end_last_path=paths[3],
        base_prompt="cinematic scene",
        extender=extender,
    )
    assert prompt == "smooth transition prompt"


def test_join_vlm_prompts_returns_none_for_missing_quads():
    result = join_vlm_enhancement._generate_vlm_prompts_for_joins(
        image_quads=[(None, None, None, None)],
        base_prompt="base",
        vlm_device="cpu",
    )
    assert result == [None]


def test_join_vlm_prompts_delegate_to_shared_service(monkeypatch):
    called = {}

    def _fake_service(**kwargs):
        called.update(kwargs)
        return [None, "join service prompt"]

    monkeypatch.setattr(join_vlm_enhancement, "_service_generate_join_quad_prompts", _fake_service)

    result = join_vlm_enhancement._generate_vlm_prompts_for_joins(
        image_quads=[
            (None, None, None, None),
            ("a.png", "b.png", "c.png", "d.png"),
        ],
        base_prompt="base",
        vlm_device="cpu",
    )

    assert result == [None, "join service prompt"]
    assert called == {
        "image_quads": [
            (None, None, None, None),
            ("a.png", "b.png", "c.png", "d.png"),
        ],
        "base_prompt": "base",
        "device": "cpu",
    }


def test_edit_video_keeper_segments_and_clip_path():
    keepers = edit_video_orchestrator._calculate_keeper_segments(
        portions=[{"start_frame": 2, "end_frame": 3}, {"start_frame": 6, "end_frame": 7}],
        total_frames=10,
        replace_mode=False,
    )
    assert keepers[0] == {"start_frame": 0, "end_frame": 1, "frame_count": 2}
    assert keepers[1] == {"start_frame": 4, "end_frame": 5, "frame_count": 2}
    assert keepers[2] == {"start_frame": 8, "end_frame": 9, "frame_count": 2}

    clip_path = edit_video_orchestrator._get_clip_url_or_path(
        local_path=Path("/mock/clip.mp4"),
        project_id=None,
        task_id="task-clip",
        clip_name="clip-1",
    )
    assert clip_path == "/mock/clip.mp4"


def test_magic_edit_returns_error_when_replicate_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(magic_edit, "replicate", None)
    reported = []
    monkeypatch.setattr(magic_edit, "report_orchestrator_failure", lambda *args, **kwargs: reported.append(True))

    ok, message = magic_edit.handle_magic_edit_task({}, tmp_path, "magic-1")
    assert ok is False
    assert "Replicate library not installed" in message
    assert reported


def test_travel_guide_prepare_vace_ref_skips_when_no_original_path(tmp_path):
    result = travel_guide.prepare_vace_ref_for_segment(
        ref_instruction={},
        segment_processing_dir=tmp_path,
        target_resolution_wh=(512, 512),
        task_id_for_logging="travel-1",
    )
    assert result is None


def test_qwen_handler_falls_back_to_default_variant(tmp_path):
    handler = qwen_handler.QwenHandler(wan_root=str(tmp_path), task_id="qwen-1")
    model_name = handler.get_edit_model_name({"qwen_edit_model": "unknown-variant"})
    assert model_name == "qwen_image_edit_20B"

    params = {}
    handler._ensure_lora_lists(params)
    assert params["lora_names"] == []
    assert params["lora_multipliers"] == []
