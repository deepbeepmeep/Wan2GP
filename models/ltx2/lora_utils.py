import os


def _is_nonzero_multiplier(value) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_is_nonzero_multiplier(item) for item in value)
    try:
        return abs(float(value)) > 1e-8
    except (TypeError, ValueError):
        return False


def is_ic_lora_filename(value) -> bool:
    return "ic-lora" in os.path.basename(str(value)).lower()


def phase2_ic_lora_name(loras_selected, loras_slists) -> str | None:
    phase2 = (loras_slists or {}).get("phase2", [])
    for index, lora_path in enumerate(loras_selected or []):
        name = os.path.basename(str(lora_path))
        if not is_ic_lora_filename(name):
            continue
        if not loras_slists or index < len(phase2) and _is_nonzero_multiplier(phase2[index]):
            return name
    return None


def control_video_phase2_message(loras_selected, loras_slists) -> str:
    lora_name = phase2_ic_lora_name(loras_selected, loras_slists)
    if lora_name is not None:
        return f'Control Video will also be injected in LTX-2 Phase 2 since a non null phase 2 lora multiplier has been detected for Ic Lora "{lora_name}"'
    return "Control Video will be only injected in LTX-2 Phase 1 since there isnt't any non null phase 2 Ic Lora multiplier"
