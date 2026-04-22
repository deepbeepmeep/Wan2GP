from source.task_handlers.tasks.task_registry import _inject_ic_lora_with_dedup


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
