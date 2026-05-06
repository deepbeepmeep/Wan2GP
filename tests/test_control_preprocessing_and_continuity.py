from __future__ import annotations

import numpy as np
import pytest

from source.media.structure import preprocessors
from source.task_handlers.tasks.template_routing import (
    WorkerBackend,
    derive_route_key,
    resolve_task_route,
)


def _frames(count: int = 3) -> list[np.ndarray]:
    return [
        np.full((2, 2, 3), idx * 10, dtype=np.uint8)
        for idx in range(count)
    ]


def test_canny_depth_pose_raw_flow_and_uni3c_preprocessors_preserve_frame_count_and_modifiers(monkeypatch):
    frames = _frames(3)
    monkeypatch.setattr(preprocessors.Path, "exists", lambda _self: True)

    class _Canny:
        def __init__(self, _cfg):
            pass

        def forward(self, input_frames):
            return [np.full_like(frame, 100) for frame in input_frames]

    class _Depth:
        def __init__(self, _cfg):
            pass

        def forward(self, input_frames):
            return [np.full_like(frame, 100) for frame in input_frames]

    class _Pose:
        def __init__(self, _cfg):
            pass

        def forward(self, input_frames):
            return [frame + 1 for frame in input_frames]

    class _Flow:
        def __init__(self, _cfg):
            pass

        def forward(self, input_frames):
            return [np.ones_like(input_frames[0]) * 2, np.ones_like(input_frames[0]) * 4], None

    class _FlowViz:
        @staticmethod
        def flow_to_image(flow):
            return flow.astype(np.uint8)

    monkeypatch.setattr(preprocessors, "get_canny_video_annotator_class", lambda: _Canny)
    monkeypatch.setattr(preprocessors, "get_depth_v2_video_annotator_class", lambda: _Depth)
    monkeypatch.setattr(preprocessors, "get_pose_body_face_video_annotator_class", lambda: _Pose)
    monkeypatch.setattr(preprocessors, "get_flow_annotator_class", lambda: _Flow)
    monkeypatch.setattr(preprocessors, "get_flow_viz_module", lambda: _FlowViz)

    canny = preprocessors.process_structure_frames(
        frames, "canny", motion_strength=1.0, canny_intensity=1.5, depth_contrast=1.0
    )
    depth = preprocessors.process_structure_frames(
        frames, "depth", motion_strength=1.0, canny_intensity=1.0, depth_contrast=2.0
    )
    pose = preprocessors.process_structure_frames(
        frames, "pose", motion_strength=1.0, canny_intensity=1.0, depth_contrast=1.0
    )
    flow = preprocessors.process_structure_frames(
        frames, "flow", motion_strength=0.5, canny_intensity=1.0, depth_contrast=1.0
    )
    raw = preprocessors.process_structure_frames(
        frames, "raw", motion_strength=1.0, canny_intensity=1.0, depth_contrast=1.0
    )
    uni3c = preprocessors.process_structure_frames(
        frames, "uni3c", motion_strength=1.0, canny_intensity=1.0, depth_contrast=1.0
    )

    assert len(canny) == len(frames)
    assert int(canny[0][0, 0, 0]) == 150
    assert len(depth) == len(frames)
    assert int(depth[0][0, 0, 0]) == 72
    assert len(pose) == len(frames)
    assert np.array_equal(pose[1], frames[1] + 1)
    assert len(flow) == len(frames)
    assert [int(frame[0, 0, 0]) for frame in flow] == [1, 1, 2]
    assert raw is frames
    assert uni3c is frames


def test_preprocessor_count_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(
        preprocessors,
        "get_structure_preprocessor",
        lambda *_args, **_kwargs: lambda input_frames: input_frames[:1],
    )

    with pytest.raises(ValueError, match="returned 1 frames for 3 input frames"):
        preprocessors.process_structure_frames(
            _frames(3), "canny", motion_strength=1.0, canny_intensity=1.0, depth_contrast=1.0
        )


@pytest.mark.parametrize(
    ("params", "expected_continuity"),
    [
        ({"model_name": "wan_2_2_i2v_lightning_baseline_2_2_2"}, "first_last"),
        ({"model_name": "wan_2_2_i2v_lightning_baseline_2_2_2", "video_source": "/tmp/prefix.mp4"}, "video_source"),
        ({"model_name": "wan_2_2_i2v_lightning_baseline_2_2_2", "svi2pro": True, "continuity_case": "svi"}, "svi"),
        ({"model_name": "wan_2_2_vace_lightning_baseline_2_2_2", "continuity_case": "join_bridge"}, "join_bridge"),
        ({"model_name": "ltx2_22B", "independent_segments": True}, "first_last"),
    ],
)
def test_continuity_cases_are_reflected_in_route_keys(params, expected_continuity):
    route_key = derive_route_key("travel_segment", params)

    assert f"continuity-{expected_continuity}" in route_key


def test_standalone_regeneration_fields_drive_video_source_continuity_route():
    route_key = derive_route_key(
        "individual_travel_segment",
        {
            "model_name": "ltx2_22B_distilled",
            "travel_guidance": {"kind": "ltx_anchor"},
            "video_source": "/tmp/regeneration-prefix.mp4",
            "override_profile": 3,
            "child_generation_id": "child-1",
            "pair_shot_generation_id": "pair-1",
        },
    )

    assert route_key == (
        "individual_travel_segment__model-ltx2_distilled__guidance-ltx_anchor"
        "__continuity-video_source__profile-3"
    )


@pytest.mark.parametrize("mode", ["pose", "depth", "canny", "cameraman"])
def test_ltx_control_rows_requiring_control_guides_do_not_use_first_last_template(mode):
    resolved = resolve_task_route(
        task_id=f"ltx-control-{mode}",
        task_type="travel_segment",
        params={
            "model_name": "ltx2_22B_distilled",
            "continuity_case": "first_last",
            "travel_guidance": {
                "kind": "ltx_control",
                "mode": mode,
                "videos": [{"path": f"/tmp/{mode}.mp4"}],
            },
        },
        backend="vibecomfy",
    )

    assert resolved.backend is WorkerBackend.VIBECOMFY
    assert resolved.template_id is None
    assert resolved.fail_closed_reason
    assert "will not fall back to WGP" in resolved.fail_closed_reason
