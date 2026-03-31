from pathlib import Path
from unittest.mock import patch

import pytest

from source.core.params.structure_guidance import StructureGuidanceConfig, StructureVideoEntry
from source.core.params.travel_guidance import TravelGuidanceConfig
from source.models.wgp.generation_helpers import is_t2v
from source.task_handlers.tasks.task_registry import _apply_uni3c_config
from source.task_handlers.tasks.task_registry import SegmentContext, GenerationInputs, StructureOutputs
from source.task_handlers.travel.guide_builder import create_guide_video
from source.task_handlers.travel.orchestrator import (
    IC_LORA_UNION_CONTROL_FILENAME,
    _auto_inject_travel_guidance_lora,
    _build_segment_travel_guidance_payload,
    _calculate_segment_stitched_offsets,
    _segment_has_travel_guidance_overlap,
)
from source.task_handlers.travel.segment_processor import TravelSegmentContext, TravelSegmentProcessor


VIDEO_ENTRY = {
    "path": "/tmp/guidance.mp4",
    "start_frame": 0,
    "end_frame": 16,
    "treatment": "adjust",
}


class _T2VProbe:
    def __init__(self, model_name: str):
        self.current_model = model_name

    def _get_base_model_type(self, model_name: str) -> str:
        return model_name.removesuffix("_distilled")


def _make_processor(mode: str, *, model_name: str = "ltx2_22B_distilled") -> TravelSegmentProcessor:
    ctx = TravelSegmentContext(
        task_id="task-1",
        segment_idx=0,
        model_name=model_name,
        total_frames_for_segment=17,
        parsed_res_wh=(64, 64),
        segment_processing_dir=Path("/tmp"),
        main_output_dir_base=Path("/tmp"),
        orchestrator_details={"guide_preprocessing": "PVG", "fps_helpers": 16},
        segment_params={
            "travel_guidance": {
                "kind": "ltx_control",
                "mode": mode,
                "videos": [VIDEO_ENTRY],
            }
        },
        mask_active_frames=False,
        debug_enabled=False,
    )
    return TravelSegmentProcessor(ctx)


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("t2v", True),
        ("t2v_1.3B", True),
        ("hunyuan", True),
        ("ltxv_13B", True),
        ("ltx2_19B", True),
        ("ltx2_22B", True),
        ("ltx2_22B_distilled", True),
        ("flux", False),
    ],
)
def test_is_t2v_recognizes_supported_base_model_types(model_name, expected):
    assert is_t2v(_T2VProbe(model_name)) is expected


@pytest.mark.parametrize("mode", ["pose", "depth", "canny", "video"])
def test_vpt_is_vg_for_all_ltx_control_modes(mode):
    processor = _make_processor(mode)
    assert processor.create_video_prompt_type(mask_video_path=None) == "VG"


def test_guide_preprocessing_ignored_when_travel_guidance_present():
    processor = _make_processor("pose")
    assert processor.create_video_prompt_type(mask_video_path=None) == "VG"


@pytest.mark.parametrize("mode", ["pose", "depth", "canny"])
def test_ic_lora_injected_for_control_modes(mode):
    config = TravelGuidanceConfig.from_payload(
        {"kind": "ltx_control", "mode": mode, "videos": [VIDEO_ENTRY]},
        "ltx2_22B_distilled",
    )

    loras = _auto_inject_travel_guidance_lora([], config)

    assert loras == [
        {
            "path": IC_LORA_UNION_CONTROL_FILENAME,
            "strength": config.strength,
            "name": "ic-lora-union-control (auto-injected)",
        }
    ]


def test_ic_lora_not_injected_for_video_mode():
    config = TravelGuidanceConfig.from_payload(
        {"kind": "ltx_control", "mode": "video", "videos": [VIDEO_ENTRY]},
        "ltx2_22B_distilled",
    )

    assert _auto_inject_travel_guidance_lora([], config) == []


def test_uni3c_prompt_exclusion_still_works():
    ctx = TravelSegmentContext(
        task_id="task-uni3c",
        segment_idx=0,
        model_name="wan_2_2_vace_lightning_baseline_2_2_2",
        total_frames_for_segment=17,
        parsed_res_wh=(64, 64),
        segment_processing_dir=Path("/tmp"),
        main_output_dir_base=Path("/tmp"),
        orchestrator_details={},
        segment_params={
            "travel_guidance": {
                "kind": "uni3c",
                "videos": [VIDEO_ENTRY],
            }
        },
        mask_active_frames=False,
        debug_enabled=False,
    )
    processor = TravelSegmentProcessor(ctx)
    processor._structure_config = StructureGuidanceConfig(
        target="uni3c",
        videos=[StructureVideoEntry.from_dict(VIDEO_ENTRY)],
    )
    processor._travel_guidance_config = TravelGuidanceConfig.from_payload(
        {"kind": "uni3c", "videos": [VIDEO_ENTRY]},
        "wan_2_2_vace_lightning_baseline_2_2_2",
    )

    assert processor.create_video_prompt_type(mask_video_path=None) == ""


def test_segment_frame_offsets_correct_for_multi_segment_ltx_control():
    config = TravelGuidanceConfig.from_payload(
        {
            "kind": "ltx_control",
            "mode": "pose",
            "videos": [{"path": "/tmp/a.mp4", "start_frame": 10, "end_frame": 18, "treatment": "adjust"}],
        },
        "ltx2_22B_distilled",
    )
    segment_frames = [9, 9, 9]
    overlaps = [4, 4]
    stitched_offsets, total_stitched = _calculate_segment_stitched_offsets(segment_frames, overlaps)

    segment_has_guidance = _segment_has_travel_guidance_overlap(
        segment_index=2,
        segment_frames_expanded=segment_frames,
        segment_stitched_offsets=stitched_offsets,
        total_stitched_frames=total_stitched,
        travel_guidance_config=config,
    )
    payload = _build_segment_travel_guidance_payload(
        config,
        frame_offset=stitched_offsets[2],
        has_guidance=segment_has_guidance,
    )

    assert stitched_offsets == [0, 5, 10]
    assert payload["_frame_offset"] == 10


def test_no_coverage_segment_gets_kind_none():
    config = TravelGuidanceConfig.from_payload(
        {
            "kind": "ltx_control",
            "mode": "pose",
            "videos": [{"path": "/tmp/a.mp4", "start_frame": 0, "end_frame": 4, "treatment": "adjust"}],
        },
        "ltx2_22B_distilled",
    )
    segment_frames = [9, 9]
    overlaps = [4]
    stitched_offsets, total_stitched = _calculate_segment_stitched_offsets(segment_frames, overlaps)

    segment_has_guidance = _segment_has_travel_guidance_overlap(
        segment_index=1,
        segment_frames_expanded=segment_frames,
        segment_stitched_offsets=stitched_offsets,
        total_stitched_frames=total_stitched,
        travel_guidance_config=config,
    )
    payload = _build_segment_travel_guidance_payload(
        config,
        frame_offset=stitched_offsets[1],
        has_guidance=segment_has_guidance,
    )

    assert payload == {"kind": "none"}


def test_ltx_control_segment_bypasses_rife_guide_builder(tmp_path):
    ctx = TravelSegmentContext(
        task_id="task-bypass",
        segment_idx=0,
        model_name="ltx2_22B_distilled",
        total_frames_for_segment=17,
        parsed_res_wh=(64, 64),
        segment_processing_dir=tmp_path,
        main_output_dir_base=tmp_path,
        orchestrator_details={"fps_helpers": 16},
        segment_params={
            "travel_guidance": {
                "kind": "ltx_control",
                "mode": "pose",
                "videos": [VIDEO_ENTRY],
                "_guidance_video_url": "/tmp/guidance.mp4",
                "_frame_offset": 0,
            }
        },
        mask_active_frames=False,
        debug_enabled=False,
    )
    processor = TravelSegmentProcessor(ctx)
    direct_path = tmp_path / "direct.mp4"
    direct_path.write_bytes(b"video")

    with patch("source.task_handlers.travel.guide_builder._create_ltx_control_guide_video", return_value=direct_path) as direct_mock:
        with patch("source.task_handlers.travel.guide_builder.create_guide_video_for_travel_segment", side_effect=AssertionError("RIFE path should not run")):
            result = create_guide_video(processor)

    direct_mock.assert_called_once()
    assert result == direct_path


def test_ic_lora_strength_is_used_as_multiplier():
    config = TravelGuidanceConfig.from_payload(
        {
            "kind": "ltx_control",
            "mode": "pose",
            "videos": [VIDEO_ENTRY],
            "strength": 0.8,
        },
        "ltx2_22B_distilled",
    )

    loras = _auto_inject_travel_guidance_lora([], config)
    assert loras[0]["strength"] == 0.8


def test_auto_inject_lora_does_not_mutate_original_entries():
    """Ensure shared LoRA dicts from loras_per_segment_expanded are not mutated."""
    original_lora = {"path": IC_LORA_UNION_CONTROL_FILENAME, "strength": 1.0}
    shared_list = [original_lora]

    config = TravelGuidanceConfig.from_payload(
        {"kind": "ltx_control", "mode": "pose", "videos": [VIDEO_ENTRY], "strength": 0.3},
        "ltx2_22B_distilled",
    )

    result = _auto_inject_travel_guidance_lora(shared_list, config)

    assert result[0]["strength"] == 0.3
    assert original_lora["strength"] == 1.0  # original must be untouched
