from source.core.params.travel_guidance import TravelGuidanceConfig, _IC_LORA_UNION_CONTROL


VIDEO_ENTRY = {
    "path": "/tmp/guidance.mp4",
    "start_frame": 0,
    "end_frame": 16,
    "treatment": "adjust",
}


def test_get_ic_lora_entry_cameraman_returns_cseti_url():
    config = TravelGuidanceConfig(
        kind="ltx_control",
        mode="cameraman",
        strength=0.7,
        videos=[VIDEO_ENTRY],
    )

    entry = config.get_ic_lora_entry()

    assert entry is not None
    assert entry["path"] == "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v1/resolve/main/LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors"
    assert entry["name"] == "ic-lora-cameraman (auto-injected)"
    assert entry["strength"] == 0.7


def test_get_ic_lora_entry_pose_falls_back_to_union_control():
    config = TravelGuidanceConfig(
        kind="ltx_control",
        mode="pose",
        strength=0.7,
        videos=[VIDEO_ENTRY],
    )

    entry = config.get_ic_lora_entry()

    assert entry is not None
    assert entry["path"] == _IC_LORA_UNION_CONTROL
