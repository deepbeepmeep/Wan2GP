import pytest

from source.core.params.structure_guidance import StructureGuidanceConfig
from source.core.params.travel_guidance import TravelGuidanceConfig


VIDEO_ENTRY = {
    "path": "/tmp/guidance.mp4",
    "start_frame": 0,
    "end_frame": 16,
    "treatment": "adjust",
}


def test_parse_each_travel_guidance_kind():
    vace = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]}},
        "wan_2_2_vace_lightning_baseline_2_2_2",
    )
    ltx = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": [VIDEO_ENTRY]}},
        "ltx2_22B_distilled",
    )
    uni3c = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "uni3c", "videos": [VIDEO_ENTRY]}},
        "ltx2_22B_distilled",
    )
    none_cfg = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "none"}},
        "ltx2_22B",
    )

    assert vace.kind == "vace"
    assert ltx.kind == "ltx_control"
    assert uni3c.kind == "uni3c"
    assert none_cfg.kind == "none"


@pytest.mark.parametrize(
    ("payload", "model_name"),
    [
        ({"travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]}}, "ltx2_22B_distilled"),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": [VIDEO_ENTRY]}}, "wan_2_2_vace_lightning_baseline_2_2_2"),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": [VIDEO_ENTRY]}}, "ltx2_22B"),
    ],
)
def test_model_compatibility_errors(payload, model_name):
    with pytest.raises(ValueError):
        TravelGuidanceConfig.from_payload(payload, model_name)


@pytest.mark.parametrize("mode", ["flow", "canny", "depth", "raw"])
def test_to_structure_guidance_config_round_trip_for_vace(mode):
    config = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "vace", "mode": mode, "videos": [VIDEO_ENTRY]}},
        "wan_2_2_vace_lightning_baseline_2_2_2",
    )

    structure_config = config.to_structure_guidance_config()

    assert isinstance(structure_config, StructureGuidanceConfig)
    assert structure_config.target == "vace"
    expected_preprocessing = "none" if mode == "raw" else mode
    assert structure_config.preprocessing == expected_preprocessing


def test_to_structure_guidance_config_round_trip_for_uni3c():
    config = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "uni3c", "videos": [VIDEO_ENTRY]}},
        "ltx2_22B_distilled",
    )

    structure_config = config.to_structure_guidance_config()

    assert structure_config.target == "uni3c"
    assert structure_config.preprocessing == "none"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("pose", True), ("depth", True), ("canny", True), ("video", False)],
)
def test_needs_ic_lora(mode, expected):
    config = TravelGuidanceConfig.from_payload(
        {"travel_guidance": {"kind": "ltx_control", "mode": mode, "videos": [VIDEO_ENTRY]}},
        "ltx2_22B_distilled",
    )
    assert config.needs_ic_lora() is expected


@pytest.mark.parametrize(
    ("payload", "model_name", "expected"),
    [
        ({"travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]}}, "wan_2_2_vace_lightning_baseline_2_2_2", 1.0),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": [VIDEO_ENTRY]}}, "ltx2_22B_distilled", 0.5),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "video", "videos": [VIDEO_ENTRY]}}, "ltx2_22B_distilled", 1.0),
    ],
)
def test_strength_defaults(payload, model_name, expected):
    config = TravelGuidanceConfig.from_payload(payload, model_name)
    assert config.strength == expected


def test_exclusivity_rejects_structure_guidance():
    with pytest.raises(ValueError):
        TravelGuidanceConfig.from_payload(
            {
                "travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]},
                "structure_guidance": {"target": "vace"},
            },
            "wan_2_2_vace_lightning_baseline_2_2_2",
        )

    with pytest.raises(ValueError):
        TravelGuidanceConfig.from_payload(
            {
                "travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]},
                "structure_type": "flow",
            },
            "wan_2_2_vace_lightning_baseline_2_2_2",
        )


def test_exclusivity_allows_falsy_legacy_fields():
    """use_uni3c: false and empty legacy fields should NOT conflict."""
    config = TravelGuidanceConfig.from_payload(
        {
            "travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]},
            "use_uni3c": False,
            "structure_type": "",
            "structure_videos": [],
            "structure_video_path": None,
        },
        "wan_2_2_vace_lightning_baseline_2_2_2",
    )
    assert config.kind == "vace"


@pytest.mark.parametrize(
    ("payload", "model_name"),
    [
        ({"travel_guidance": {"kind": "vace", "mode": "flow", "videos": []}}, "wan_2_2_vace_lightning_baseline_2_2_2"),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": []}}, "ltx2_22B_distilled"),
        ({"travel_guidance": {"kind": "uni3c", "videos": []}}, "ltx2_22B_distilled"),
    ],
)
def test_empty_videos_on_non_none_kind_errors(payload, model_name):
    with pytest.raises(ValueError):
        TravelGuidanceConfig.from_payload(payload, model_name)


@pytest.mark.parametrize(
    ("payload", "model_name", "expected"),
    [
        ({"travel_guidance": {"kind": "vace", "mode": "flow", "videos": [VIDEO_ENTRY]}}, "wan_2_2_vace_lightning_baseline_2_2_2", "flow"),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "pose", "videos": [VIDEO_ENTRY]}}, "ltx2_22B_distilled", "pose"),
        ({"travel_guidance": {"kind": "ltx_control", "mode": "video", "videos": [VIDEO_ENTRY]}}, "ltx2_22B_distilled", "raw"),
    ],
)
def test_get_preprocessor_type(payload, model_name, expected):
    config = TravelGuidanceConfig.from_payload(payload, model_name)
    assert config.get_preprocessor_type() == expected
