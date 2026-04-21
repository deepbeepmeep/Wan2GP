from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from source.core.params.travel_guidance import TravelGuidanceConfig
from source.core.params.generation_policy import GenerationPolicy
from source.models.wgp.generators.preflight import prepare_svi_image_refs
from source.models.wgp.generators.wgp_params import build_normal_params
from source.task_handlers.tasks.task_conversion import db_task_to_generation_task
from source.task_handlers.tasks.task_registry import (
    GenerationInputs,
    ImageRefs,
    SegmentContext,
    StructureOutputs,
    _build_generation_params,
)
from source.task_handlers.travel.orchestrator import (
    _build_segment_anchor_guidance_config,
    _calculate_segment_stitched_offsets,
    _segment_has_travel_guidance_overlap,
)

# Pre-existing IC-LoRA helpers moved out of orchestrator (commit bc26726d);
# kept as lazy imports so the rest of this module stays collectable.
try:
    from source.task_handlers.travel.orchestrator import (  # type: ignore[attr-defined]
        IC_LORA_UNION_CONTROL_FILENAME,
        _auto_inject_travel_guidance_lora,
    )
    _IC_LORA_HELPERS_AVAILABLE = True
except ImportError:
    IC_LORA_UNION_CONTROL_FILENAME = ""  # type: ignore[assignment]
    _auto_inject_travel_guidance_lora = None  # type: ignore[assignment]
    _IC_LORA_HELPERS_AVAILABLE = False
from source.task_handlers.travel.segment_processor import (
    TravelSegmentContext,
    TravelSegmentProcessor,
)


VIDEO_ENTRY = {
    "path": "/tmp/guidance.mp4",
    "start_frame": 0,
    "end_frame": 16,
    "treatment": "adjust",
}


def _write_image(path: Path, color: str = "red") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)
    return str(path)


def _hybrid_payload(
    *,
    anchors: list[dict] | None = None,
    videos: list[dict] | None = None,
    mode: str = "",
    audio: dict | None = None,
    control_strength: float = 0.7,
    strength: float = 0.0,
) -> dict:
    payload = {
        "kind": "ltx_hybrid",
        "anchors": anchors or [],
        "videos": videos or [],
        "control_strength": control_strength,
        "strength": strength,
    }
    if mode:
        payload["mode"] = mode
    if audio is not None:
        payload["audio"] = audio
    return payload


def _make_processor(
    tmp_path: Path,
    travel_guidance: dict,
    *,
    model_name: str = "ltx2_22B_distilled",
    segment_frames: int = 17,
) -> TravelSegmentProcessor:
    ctx = TravelSegmentContext(
        task_id="task-hybrid",
        segment_idx=0,
        model_name=model_name,
        total_frames_for_segment=segment_frames,
        parsed_res_wh=(64, 64),
        segment_processing_dir=tmp_path,
        main_output_dir_base=tmp_path,
        orchestrator_details={"fps_helpers": 16},
        segment_params={"travel_guidance": travel_guidance},
        mask_active_frames=False,
        debug_enabled=False,
    )
    return TravelSegmentProcessor(ctx)


def test_hybrid_contract_parse_validate_and_round_trip(tmp_path: Path):
    anchor_a = _write_image(tmp_path / "a.png", "red")
    anchor_b = _write_image(tmp_path / "b.png", "blue")
    payload = {
        "travel_guidance": _hybrid_payload(
            anchors=[
                {"image_url": anchor_a, "frame_position": 1, "strength": 0.4},
                {"image_url": anchor_b, "frame_position": 9, "strength": 0.8},
            ],
            videos=[VIDEO_ENTRY],
            mode="pose",
            audio={"source": "external", "audio_url": str(tmp_path / "guide.wav"), "strength": 0.6},
            control_strength=0.75,
        )
    }

    config = TravelGuidanceConfig.from_payload(payload, "ltx2_22B_distilled")
    stitched_offsets, total_stitched = _calculate_segment_stitched_offsets([17, 17], [4])
    segment_config = _build_segment_anchor_guidance_config(
        config=config,
        segment_index=0,
        segment_frames_expanded=[17, 17],
        segment_stitched_offsets=stitched_offsets,
        total_stitched_frames=total_stitched,
    )
    reparsed = TravelGuidanceConfig.from_payload(
        {"travel_guidance": segment_config.to_segment_payload(frame_offset=segment_config.frame_offset)},
        "ltx2_22B_distilled",
    )

    assert config.is_ltx_hybrid
    assert config.has_guidance
    assert config.has_control
    assert config.has_anchors
    assert config.has_audio
    assert config.needs_ic_lora() is True
    assert reparsed.anchors[0].frame_position == 1
    assert reparsed.control_strength == 0.75
    assert reparsed.audio.source == "external"


@pytest.mark.parametrize(
    "payload",
    [
        _hybrid_payload(),
        _hybrid_payload(anchors=[{"image_url": "", "frame_position": 1}]),
        _hybrid_payload(
            anchors=[{"image_url": "/tmp/a.png", "frame_position": 1, "strength": 2.0}]
        ),
        _hybrid_payload(
            anchors=[{"image_url": "/tmp/a.png", "frame_position": 1}],
            audio={"source": "control_track", "strength": 1.0},
        ),
        _hybrid_payload(
            anchors=[{"image_url": "/tmp/a.png", "frame_position": 1}],
            audio={"source": "external", "strength": 1.0},
        ),
    ],
)
def test_hybrid_contract_invalid_combinations(payload: dict):
    with pytest.raises(ValueError):
        TravelGuidanceConfig.from_payload({"travel_guidance": payload}, "ltx2_22B_distilled")


def test_overlap_gating_and_anchor_zone_membership(tmp_path: Path):
    config = TravelGuidanceConfig.from_payload(
        {
            "travel_guidance": _hybrid_payload(
                anchors=[
                    {"image_url": _write_image(tmp_path / "a.png"), "frame_position": 13, "strength": 0.5}
                ]
            )
        },
        "ltx2_22B_distilled",
    )
    stitched_offsets, total_stitched = _calculate_segment_stitched_offsets([17, 17], [4])

    assert _segment_has_travel_guidance_overlap(
        segment_index=0,
        segment_frames_expanded=[17, 17],
        segment_stitched_offsets=stitched_offsets,
        total_stitched_frames=total_stitched,
        travel_guidance_config=config,
    )
    assert _segment_has_travel_guidance_overlap(
        segment_index=1,
        segment_frames_expanded=[17, 17],
        segment_stitched_offsets=stitched_offsets,
        total_stitched_frames=total_stitched,
        travel_guidance_config=config,
    )


def test_preflight_strict_alignment_fails_hard(tmp_path: Path):
    valid = _write_image(tmp_path / "valid.png")
    with pytest.raises(ValueError):
        prepare_svi_image_refs(
            {
                "image_refs_paths": [valid, str(tmp_path / "missing.png")],
                "image_refs_strengths": [1.0, 0.5],
            }
        )


def test_preflight_strict_alignment_rejects_empty_anchor_path(tmp_path: Path):
    valid = _write_image(tmp_path / "valid.png")
    with pytest.raises(ValueError):
        prepare_svi_image_refs(
            {
                "image_refs_paths": [valid, ""],
                "image_refs_strengths": [1.0, 0.5],
            }
        )


def test_preflight_legacy_mode_still_skips_bad_images(tmp_path: Path):
    valid = _write_image(tmp_path / "valid.png")
    kwargs = {"image_refs_paths": [valid, str(tmp_path / "missing.png")]}
    prepare_svi_image_refs(kwargs)
    assert len(kwargs["image_refs"]) == 1


@pytest.mark.parametrize(
    ("travel_guidance", "expected"),
    [
        (_hybrid_payload(videos=[VIDEO_ENTRY], mode="video"), "VG"),
        (_hybrid_payload(videos=[VIDEO_ENTRY], mode="pose"), "PVG"),
        (_hybrid_payload(videos=[VIDEO_ENTRY], mode="depth"), "DVG"),
        (_hybrid_payload(videos=[VIDEO_ENTRY], mode="canny"), "EVG"),
        (_hybrid_payload(anchors=[{"image_url": "/tmp/a.png", "frame_position": 0}]), "KFI"),
        (
            _hybrid_payload(
                anchors=[{"image_url": "/tmp/a.png", "frame_position": 0}],
                videos=[VIDEO_ENTRY],
                mode="pose",
            ),
            "PVGKFI",
        ),
    ],
)
def test_hybrid_prompt_type_matrix(tmp_path: Path, travel_guidance: dict, expected: str):
    processor = _make_processor(tmp_path, travel_guidance)
    assert processor.create_video_prompt_type(mask_video_path=None) == expected


def test_hybrid_process_segment_builds_parallel_anchor_lists_and_audio(tmp_path: Path):
    audio_path = tmp_path / "guide.wav"
    audio_path.write_bytes(b"audio")
    anchor_a = _write_image(tmp_path / "a.png", "red")
    anchor_b = _write_image(tmp_path / "b.png", "blue")
    guide_path = tmp_path / "guide.mp4"
    guide_path.write_bytes(b"video")

    processor = _make_processor(
        tmp_path,
        _hybrid_payload(
            anchors=[
                {"image_url": anchor_a, "frame_position": 0, "strength": 0.4},
                {"image_url": anchor_b, "frame_position": 8, "strength": 0.9},
            ],
            videos=[VIDEO_ENTRY],
            mode="pose",
            audio={"source": "external", "audio_url": str(audio_path), "strength": 0.55},
            control_strength=0.65,
        ),
    )

    def _fake_ffmpeg(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"wav")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("source.task_handlers.travel.guide_builder._create_ltx_control_guide_video", return_value=guide_path):
        with patch("source.task_handlers.travel.segment_processor.get_video_frame_count_and_fps", return_value=(17, 16)):
            with patch("source.task_handlers.travel.segment_processor.run_subprocess", side_effect=_fake_ffmpeg):
                result = processor.process_segment()

    assert result["video_prompt_type"] == "PVGKFI"
    assert result["image_refs_paths"] == [anchor_a, anchor_b]
    assert result["frames_positions"] == "1,9"
    assert result["image_refs_strengths"] == [0.4, 0.9]
    assert result["audio_prompt_type"] == "A"
    assert result["audio_scale"] == 0.55
    assert result["denoising_strength"] == 0.65
    assert Path(result["audio_guide"]).exists()


def test_hybrid_control_track_audio_sets_k_prompt(tmp_path: Path):
    guide_path = tmp_path / "guide.mp4"
    guide_path.write_bytes(b"video")
    processor = _make_processor(
        tmp_path,
        _hybrid_payload(
            anchors=[{"image_url": _write_image(tmp_path / "a.png"), "frame_position": 0}],
            videos=[VIDEO_ENTRY],
            mode="video",
            audio={"source": "control_track", "strength": 0.3},
        ),
    )

    with patch("source.task_handlers.travel.guide_builder._create_ltx_control_guide_video", return_value=guide_path):
        with patch("source.task_handlers.travel.segment_processor.get_video_frame_count_and_fps", return_value=(17, 16)):
            result = processor.process_segment()

    assert result["audio_prompt_type"] == "K"
    assert result["audio_guide"] is None


def test_segment_hybrid_config_drops_control_track_audio_without_control_overlap(tmp_path: Path):
    config = TravelGuidanceConfig.from_payload(
        {
            "travel_guidance": _hybrid_payload(
                anchors=[{"image_url": _write_image(tmp_path / "a.png"), "frame_position": 13}],
                videos=[{"path": "/tmp/guidance.mp4", "start_frame": 0, "end_frame": 12, "treatment": "adjust"}],
                mode="video",
                audio={"source": "control_track", "strength": 0.3},
            )
        },
        "ltx2_22B_distilled",
    )
    stitched_offsets, total_stitched = _calculate_segment_stitched_offsets([17, 17], [4])

    segment_config = _build_segment_anchor_guidance_config(
        config=config,
        segment_index=1,
        segment_frames_expanded=[17, 17],
        segment_stitched_offsets=stitched_offsets,
        total_stitched_frames=total_stitched,
    )

    assert segment_config.anchors[0].frame_position == 0
    assert segment_config.videos == []
    assert segment_config.audio is None


def test_hybrid_contract_allows_duplicate_global_positions_and_segment_remap_rejects(tmp_path: Path):
    config = TravelGuidanceConfig.from_payload(
        {
            "travel_guidance": _hybrid_payload(
                anchors=[
                    {"image_url": _write_image(tmp_path / "a.png", "red"), "frame_position": 13},
                    {"image_url": _write_image(tmp_path / "b.png", "blue"), "frame_position": 13},
                ]
            )
        },
        "ltx2_22B_distilled",
    )
    stitched_offsets, total_stitched = _calculate_segment_stitched_offsets([17, 17], [4])

    with pytest.raises(ValueError, match="duplicate local frame positions"):
        _build_segment_anchor_guidance_config(
            config=config,
            segment_index=1,
            segment_frames_expanded=[17, 17],
            segment_stitched_offsets=stitched_offsets,
            total_stitched_frames=total_stitched,
        )


@pytest.mark.skipif(not _IC_LORA_HELPERS_AVAILABLE, reason="IC-LoRA helpers moved out of orchestrator")
def test_ic_lora_uses_control_strength_for_hybrid_and_strength_for_legacy(tmp_path: Path):
    hybrid = TravelGuidanceConfig.from_payload(
        {
            "travel_guidance": _hybrid_payload(
                anchors=[{"image_url": _write_image(tmp_path / "a.png"), "frame_position": 0}],
                videos=[VIDEO_ENTRY],
                mode="pose",
                control_strength=0.33,
                strength=0.05,
            )
        },
        "ltx2_22B_distilled",
    )
    legacy = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": [VIDEO_ENTRY], "strength": 0.8}},
        "ltx2_22B_distilled",
    )

    hybrid_loras = _auto_inject_travel_guidance_lora([], hybrid)
    legacy_loras = _auto_inject_travel_guidance_lora([], legacy)

    assert hybrid_loras == [
        {
            "path": IC_LORA_UNION_CONTROL_FILENAME,
            "strength": 0.33,
            "name": "ic-lora-union-control (auto-injected)",
        }
    ]
    assert legacy_loras[0]["strength"] == 0.8


def test_build_generation_params_wires_hybrid_fields():
    ctx = SegmentContext(
        mode="orchestrator",
        orchestrator_details={},
        individual_params={},
        segment_idx=0,
        segment_params={},
    )
    gen = GenerationInputs(
        model_name="ltx2_22B_distilled",
        prompt_for_wgp="prompt",
        negative_prompt_for_wgp="",
        parsed_res_wh=(64, 64),
        total_frames_for_segment=17,
        current_run_base_output_dir=Path("/tmp"),
        segment_processing_dir=Path("/tmp"),
        debug_enabled=False,
        travel_mode="ltx",
        generation_policy=GenerationPolicy.from_payload({}),
    )
    structure = StructureOutputs(
        guide_video_path="/tmp/guide.mp4",
        video_prompt_type_str="PVGKFI",
        structure_config=TravelGuidanceConfig(kind="none"),
        image_refs_paths=["/tmp/a.png"],
        frames_positions="1",
        image_refs_strengths=[0.7],
        audio_guide="/tmp/audio.wav",
        audio_prompt_type="A",
        audio_scale=0.4,
        denoising_strength=0.6,
    )

    generation_params = _build_generation_params(ctx, gen, ImageRefs(), structure, "task-1")

    assert generation_params["image_refs_paths"] == ["/tmp/a.png"]
    assert generation_params["frames_positions"] == "1"
    assert generation_params["image_refs_strengths"] == [0.7]
    assert generation_params["audio_guide"] == "/tmp/audio.wav"
    assert generation_params["audio_prompt_type"] == "A"
    assert generation_params["audio_scale"] == 0.4
    assert generation_params["denoising_strength"] == 0.6


def test_wgp_normal_params_accept_image_refs_strengths_override():
    wgp_params = build_normal_params(
        state={},
        current_model="ltx2_22B_distilled",
        image_mode=0,
        resolved_params={"image_refs_strengths": [0.2, 0.8]},
        prompt="prompt",
        actual_video_length=17,
        actual_batch_size=1,
        actual_guidance=5.0,
        final_embedded_guidance=0.0,
        is_flux=False,
        video_guide=None,
        video_mask=None,
        video_prompt_type="KFI",
        control_net_weight=None,
        control_net_weight2=None,
        activated_loras=[],
        loras_multipliers_str="",
    )

    assert wgp_params["image_refs_strengths"] == [0.2, 0.8]


def test_non_travel_task_conversion_whitelists_new_fields(monkeypatch: pytest.MonkeyPatch):
    class DummyQwenHandler:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "source.task_handlers.tasks.task_conversion.QwenHandler",
        DummyQwenHandler,
    )

    task = db_task_to_generation_task(
        {
            "prompt": "prompt",
            "model": "ltx2_22B_distilled",
            "input_video_strength": 0.9,
            "image_refs_strengths": [0.2, 0.8],
            "audio_scale": 0.6,
        },
        task_id="task-1",
        task_type="z_image_turbo",
        wan2gp_path="/tmp",
    )

    assert task.parameters["input_video_strength"] == 0.9
    assert task.parameters["image_refs_strengths"] == [0.2, 0.8]
    assert task.parameters["audio_scale"] == 0.6
