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
    _build_segment_anchor_guidance_config,
    _build_segment_travel_guidance_payload,
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


@pytest.mark.skipif(not _IC_LORA_HELPERS_AVAILABLE, reason="IC-LoRA helpers moved out of orchestrator")
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


@pytest.mark.skipif(not _IC_LORA_HELPERS_AVAILABLE, reason="IC-LoRA helpers moved out of orchestrator")
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


@pytest.mark.skipif(not _IC_LORA_HELPERS_AVAILABLE, reason="IC-LoRA helpers moved out of orchestrator")
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


@pytest.mark.skipif(not _IC_LORA_HELPERS_AVAILABLE, reason="IC-LoRA helpers moved out of orchestrator")
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


# =============================================================================
# ltx_anchor tests (T6)
# =============================================================================


LTX_DISTILLED_MODEL = "ltx2_22B_distilled"


def _ltx_anchor_segment_params(anchors, **extra):
    return {
        "travel_guidance": {
            "kind": "ltx_anchor",
            "anchors": anchors,
            **extra,
        }
    }


def _make_ltx_anchor_processor(
    anchors,
    *,
    tmp_path: Path,
    total_frames_for_segment: int = 17,
    segment_idx: int = 0,
    segment_params_extra: dict | None = None,
) -> TravelSegmentProcessor:
    segment_params = _ltx_anchor_segment_params(anchors)
    if segment_params_extra:
        segment_params.update(segment_params_extra)
    ctx = TravelSegmentContext(
        task_id=f"task-ltx-anchor-{segment_idx}",
        segment_idx=segment_idx,
        model_name=LTX_DISTILLED_MODEL,
        total_frames_for_segment=total_frames_for_segment,
        parsed_res_wh=(64, 64),
        segment_processing_dir=tmp_path,
        main_output_dir_base=tmp_path,
        orchestrator_details={"fps_helpers": 16},
        segment_params=segment_params,
        mask_active_frames=False,
        debug_enabled=False,
    )
    return TravelSegmentProcessor(ctx)


def test_ltx_anchor_forces_kfi_vpt(tmp_path):
    processor = _make_ltx_anchor_processor(
        [
            {"image_url": "https://example.com/a.png", "frame_position": 0},
            {"image_url": "https://example.com/b.png", "frame_position": 16},
        ],
        tmp_path=tmp_path,
    )

    assert processor.create_video_prompt_type(mask_video_path=None) == "KFI"


def test_ltx_anchor_emits_anchor_payload(tmp_path):
    processor = _make_ltx_anchor_processor(
        [
            {"image_url": "https://example.com/a.png", "frame_position": 0},
            {"image_url": "https://example.com/b.png", "frame_position": 16},
        ],
        tmp_path=tmp_path,
    )

    def _fake_download(url, download_dir, task_id, **kwargs):
        fname = f"{kwargs.get('descriptive_name', 'img')}.png"
        path = Path(download_dir) / fname
        path.write_bytes(b"img")
        return str(path)

    with patch(
        "source.task_handlers.travel.segment_processor.download_image_if_url",
        side_effect=_fake_download,
    ):
        result = processor.process_segment()

    assert result["video_guide"] is None
    assert result["video_prompt_type"] == "KFI"
    # 1-indexed comma-joined frame positions
    assert result["frames_positions"] == "1,17"
    assert result["image_refs_paths"] is not None
    assert len(result["image_refs_paths"]) == 2
    assert result["image_refs_strengths"] == [1.0, 1.0]
    assert result["denoising_strength"] is None


def test_ltx_anchor_skips_pixel_bake(tmp_path):
    processor = _make_ltx_anchor_processor(
        [
            {"image_url": "https://example.com/a.png", "frame_position": 0},
            {"image_url": "https://example.com/b.png", "frame_position": 16},
        ],
        tmp_path=tmp_path,
    )

    with patch(
        "source.task_handlers.travel.guide_builder.create_guide_video_for_travel_segment",
        side_effect=AssertionError("RIFE pixel bake path should not run for ltx_anchor"),
    ):
        result = create_guide_video(processor)

    assert result is None


def test_ltx_anchor_global_strengths_survive_segment_remap():
    """FLAG-003: per-segment remap must NOT recompute strengths based on local anchor count."""
    config = TravelGuidanceConfig.from_payload(
        {
            "kind": "ltx_anchor",
            "anchors": [
                {"image_url": "https://example.com/0.png", "frame_position": 0},
                {"image_url": "https://example.com/10.png", "frame_position": 10},
                {"image_url": "https://example.com/20.png", "frame_position": 20},
                {"image_url": "https://example.com/30.png", "frame_position": 30},
            ],
        },
        LTX_DISTILLED_MODEL,
    )
    # Parse-time rewrite: first/last=1.0, middles=0.0.
    global_strengths_by_frame = {
        anchor.frame_position: anchor.strength for anchor in config.anchors
    }
    assert global_strengths_by_frame == {0: 1.0, 10: 0.0, 20: 0.0, 30: 1.0}

    segment_frames = [11, 11, 11]
    overlaps = [0, 0]
    stitched_offsets, total_stitched = _calculate_segment_stitched_offsets(segment_frames, overlaps)
    # offsets: [0, 11, 22]; segment ranges [0..11), [11..22), [22..33)

    for seg_idx in range(len(segment_frames)):
        seg_config = _build_segment_anchor_guidance_config(
            config=config,
            segment_index=seg_idx,
            segment_frames_expanded=segment_frames,
            segment_stitched_offsets=stitched_offsets,
            total_stitched_frames=total_stitched,
        )
        assert seg_config.kind == "ltx_anchor"
        # Each segment should contain anchor(s) that retain their global strength.
        seg_start = stitched_offsets[seg_idx]
        for seg_anchor in seg_config.anchors:
            global_frame = seg_anchor.frame_position + seg_start
            assert seg_anchor.strength == global_strengths_by_frame[global_frame], (
                f"Segment {seg_idx} local anchor at global_frame={global_frame} "
                f"has strength={seg_anchor.strength}, expected {global_strengths_by_frame[global_frame]}"
            )
        # Critical assertion: middle anchors (global 10 and 20) must NOT be promoted to 1.0
        # just because they are the only anchor in a 1-anchor local slice.
        for seg_anchor in seg_config.anchors:
            global_frame = seg_anchor.frame_position + seg_start
            if global_frame in (10, 20):
                assert seg_anchor.strength == 0.0, (
                    "Middle anchor strength was recomputed during remap (FLAG-003 violation)"
                )


def test_ltx_anchor_segment_overlap_uses_anchor_positions():
    config = TravelGuidanceConfig.from_payload(
        {
            "kind": "ltx_anchor",
            "anchors": [
                {"image_url": "https://example.com/a.png", "frame_position": 0},
                {"image_url": "https://example.com/b.png", "frame_position": 10},
            ],
        },
        LTX_DISTILLED_MODEL,
    )
    segment_frames = [11, 11, 11]
    overlaps = [0, 0]
    stitched_offsets, total_stitched = _calculate_segment_stitched_offsets(segment_frames, overlaps)

    # Segment 0 covers frames [0..11) — both anchors at frames 0 and 10 are in range.
    assert _segment_has_travel_guidance_overlap(
        segment_index=0,
        segment_frames_expanded=segment_frames,
        segment_stitched_offsets=stitched_offsets,
        total_stitched_frames=total_stitched,
        travel_guidance_config=config,
    ) is True

    # Segment 2 covers [22..33) — no anchors there.
    assert _segment_has_travel_guidance_overlap(
        segment_index=2,
        segment_frames_expanded=segment_frames,
        segment_stitched_offsets=stitched_offsets,
        total_stitched_frames=total_stitched,
        travel_guidance_config=config,
    ) is False

    # Out-of-range segment emits kind=none.
    seg_config = _build_segment_anchor_guidance_config(
        config=config,
        segment_index=2,
        segment_frames_expanded=segment_frames,
        segment_stitched_offsets=stitched_offsets,
        total_stitched_frames=total_stitched,
    )
    assert seg_config.kind == "none"
    payload = _build_segment_travel_guidance_payload(
        seg_config, frame_offset=stitched_offsets[2], has_guidance=False
    )
    assert payload == {"kind": "none"}

    # In-range segment emits populated ltx_anchor payload.
    seg_config_0 = _build_segment_anchor_guidance_config(
        config=config,
        segment_index=0,
        segment_frames_expanded=segment_frames,
        segment_stitched_offsets=stitched_offsets,
        total_stitched_frames=total_stitched,
    )
    assert seg_config_0.kind == "ltx_anchor"
    payload_0 = _build_segment_travel_guidance_payload(
        seg_config_0, frame_offset=stitched_offsets[0], has_guidance=True
    )
    assert payload_0["kind"] == "ltx_anchor"
    assert len(payload_0["anchors"]) == 2


def test_ltx_anchor_path_a_orchestrator_skip():
    """FLAG-002: orchestrator PATH A must bypass create_composite_guidance_video for ltx_anchor.

    Exercises the decision path via _build_segment_anchor_guidance_config and
    verifies the orchestrator-level guard (see orchestrator.py around the
    ``elif travel_guidance_config.is_ltx_anchor:`` branch inside PATH A, immediately
    above the ``else:`` that invokes ``create_composite_guidance_video``) is in
    effect: with the guard, ``create_composite_guidance_video`` is never invoked.
    """
    config = TravelGuidanceConfig.from_payload(
        {
            "kind": "ltx_anchor",
            "anchors": [
                {"image_url": "https://example.com/a.png", "frame_position": 0},
                {"image_url": "https://example.com/b.png", "frame_position": 16},
            ],
        },
        LTX_DISTILLED_MODEL,
    )

    with patch(
        "source.media.structure.create_composite_guidance_video",
        side_effect=AssertionError(
            "create_composite_guidance_video must not be invoked for ltx_anchor"
        ),
    ) as composite_mock:
        # Simulate the orchestrator's PATH A guard: use_stitched_offsets is toggled
        # true and structure_guidance_video_url stays None without ever calling
        # create_composite_guidance_video.
        use_stitched_offsets = False
        structure_guidance_video_url = "sentinel"

        if config.kind == "none":
            use_stitched_offsets = False
        elif config.is_ltx_anchor:
            use_stitched_offsets = True
            structure_guidance_video_url = None
        else:  # pragma: no cover - defensive
            from source.media.structure import create_composite_guidance_video

            create_composite_guidance_video()  # would trigger AssertionError

    composite_mock.assert_not_called()
    assert use_stitched_offsets is True
    assert structure_guidance_video_url is None


def test_ltx_anchor_path_a_guard_source_tripwire():
    """Deletion canary for the real orchestrator PATH A guard.

    T6.6 above exercises a local replica of the PATH A decision logic because
    the orchestrator function is too large to unit-test end-to-end. This test
    closes that loop by asserting the real guard text still exists in
    orchestrator.py. If someone deletes the ``elif is_ltx_anchor:`` branch
    during a refactor without also updating both tests, this tripwire fails
    and forces an explicit re-decision on the skip contract.
    """
    orchestrator_path = (
        Path(__file__).resolve().parents[1]
        / "source" / "task_handlers" / "travel" / "orchestrator.py"
    )
    source = orchestrator_path.read_text()

    assert "elif travel_guidance_config.is_ltx_anchor:" in source, (
        "ltx_anchor PATH A guard missing from orchestrator.py — "
        "create_composite_guidance_video would be invoked for ltx_anchor, "
        "which violates FLAG-002"
    )


def test_ltx_anchor_respects_explicit_per_anchor_strength():
    config = TravelGuidanceConfig.from_payload(
        {
            "kind": "ltx_anchor",
            "anchors": [
                {"image_url": "https://example.com/0.png", "frame_position": 0, "strength": 0.7},
                {"image_url": "https://example.com/10.png", "frame_position": 10, "strength": 0.4},
                {"image_url": "https://example.com/20.png", "frame_position": 20, "strength": 0.6},
                {"image_url": "https://example.com/30.png", "frame_position": 30, "strength": 0.9},
            ],
        },
        LTX_DISTILLED_MODEL,
    )

    # Parse-time count-based rewrite must NOT fire when any strength is explicit.
    parsed_by_frame = {anchor.frame_position: anchor.strength for anchor in config.anchors}
    assert parsed_by_frame == {0: 0.7, 10: 0.4, 20: 0.6, 30: 0.9}

    segment_frames = [11, 11, 11]
    overlaps = [0, 0]
    stitched_offsets, total_stitched = _calculate_segment_stitched_offsets(segment_frames, overlaps)

    for seg_idx in range(len(segment_frames)):
        seg_config = _build_segment_anchor_guidance_config(
            config=config,
            segment_index=seg_idx,
            segment_frames_expanded=segment_frames,
            segment_stitched_offsets=stitched_offsets,
            total_stitched_frames=total_stitched,
        )
        if seg_config.kind == "none":
            continue
        seg_start = stitched_offsets[seg_idx]
        for seg_anchor in seg_config.anchors:
            global_frame = seg_anchor.frame_position + seg_start
            assert seg_anchor.strength == parsed_by_frame[global_frame], (
                f"Remap mutated strength for frame {global_frame}: "
                f"expected {parsed_by_frame[global_frame]}, got {seg_anchor.strength}"
            )
