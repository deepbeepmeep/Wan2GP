from __future__ import annotations

import math
import re


WGP_SLASH_COMMANDS = {"duration", "overlap", "new_shot", "loras_mult"}
SLASH_BLOCK_RE = re.compile(r"\[\s*/\s*([^\]]+?)\s*\]", re.IGNORECASE)


def has_slash_commands(prompts: list[str]) -> bool:
    return any(SLASH_BLOCK_RE.search(prompt or "") is not None for prompt in prompts)


def normalize_frame_count(frame_count: int, minimum: int, step: int, offset: int = 1) -> int:
    frame_count = max(minimum, frame_count)
    step = max(1, step)
    offset = max(0, offset)
    return math.ceil(max(0, frame_count - offset) / step) * step + offset if step > 1 else frame_count


def floor_frame_count(frame_count: int, minimum: int, step: int, offset: int = 1) -> int:
    frame_count = max(minimum, frame_count)
    step = max(1, step)
    offset = max(0, offset)
    if step <= 1:
        return frame_count
    lower = ((frame_count - offset) // step) * step + offset
    return lower if lower >= minimum else normalize_frame_count(minimum, minimum, step, offset)


def normalize_output_frame_count(frame_count: int, minimum: int, step: int, offset: int = 1) -> int:
    frame_count = max(minimum, frame_count)
    step = max(1, step)
    if step <= 1:
        return frame_count
    lower = floor_frame_count(frame_count, minimum, step, offset)
    upper = normalize_frame_count(frame_count, minimum, step, offset)
    return lower if frame_count - lower <= upper - frame_count else upper


def normalize_overlap(frame_count: int, step: int, offset: int = 1) -> tuple[int | None, str | None]:
    if frame_count < 0:
        return None, "/overlap must be 0 or a positive frame count."
    if frame_count == 0:
        return 0, None
    step = max(1, step)
    offset = max(0, offset)
    overlap = ((frame_count - offset + step // 2) // step) * step + offset
    return max(step if offset == 0 else offset, overlap), None


def _parse_duration(raw_value: str, *, fps: float, total_frames: int) -> tuple[int | None, str | None]:
    value = str(raw_value or "").strip().lower()
    try:
        if value.endswith("%"):
            frames = int(round(float(value[:-1].strip()) * float(total_frames) / 100.0))
        elif value.endswith("s"):
            frames = int(round(float(value[:-1].strip()) * float(fps)))
        else:
            frames = int(value)
    except Exception:
        return None, f"Invalid /duration value '{raw_value}'. Use frames, seconds like 5s, or a percentage like 20%."
    if frames <= 0:
        return None, "/duration must be a positive frame count."
    return frames, None


def _parse_options(prompt: str, *, supported_model_commands: set[str], allow_new_shot: bool, fps: float, total_frames: int, step: int, overlap_offset: int, default_overlap: int) -> tuple[str, dict, dict, bool, str | None]:
    wgp_options: dict = {}
    model_options: dict = {}
    has_options = False
    error = None

    def replace(match):
        nonlocal has_options, error
        has_options = True
        raw_options = []
        for part in match.group(1).split(","):
            option = part.strip()
            normalized = option[1:].strip() if option.startswith("/") else option
            key, separator, _ = normalized.partition("=")
            key = key.strip().lower()
            if raw_options and not option.startswith("/") and not separator and key not in WGP_SLASH_COMMANDS and key not in supported_model_commands:
                raw_options[-1] = f"{raw_options[-1]},{option}"
            else:
                raw_options.append(option)
        for raw_option in raw_options:
            if error is not None:
                break
            option = raw_option.strip()
            if option.startswith("/"):
                option = option[1:].strip()
            key, separator, raw_value = option.partition("=")
            key = key.strip().lower()
            value = raw_value.strip()
            if not key:
                continue
            if key == "duration":
                if not separator or not value:
                    error = "/duration requires a value, e.g. [/duration=5s]."
                    continue
                wgp_options["duration_frames"], error = _parse_duration(value, fps=fps, total_frames=total_frames)
            elif key == "overlap":
                if separator and not value:
                    error = "/overlap value cannot be empty. Use [/overlap] or [/overlap=9]."
                    continue
                try:
                    overlap_value = default_overlap if not separator else int(value)
                except Exception:
                    error = f"Invalid /overlap value '{value}'. Use an integer frame count."
                    continue
                overlap, error = normalize_overlap(overlap_value, step, overlap_offset)
                if error is not None:
                    continue
                if separator and overlap == 0 and not allow_new_shot:
                    error = "/overlap=0 is only supported by text-to-video capable models."
                    continue
                wgp_options["overlap_frames"] = overlap
                if overlap == 0:
                    wgp_options["new_shot"] = True
            elif key == "new_shot":
                if separator:
                    error = "/new_shot does not take a value."
                    continue
                if not allow_new_shot:
                    error = "/new_shot is only supported by text-to-video capable models."
                    continue
                wgp_options["overlap_frames"] = 0
                wgp_options["new_shot"] = True
            elif key == "loras_mult":
                if not separator or not value:
                    error = "/loras_mult requires a value, e.g. [/loras_mult=1;3]."
                    continue
                wgp_options["loras_multipliers"] = value
            elif key in supported_model_commands:
                model_options[key] = value if separator else True
            else:
                supported = sorted(WGP_SLASH_COMMANDS | supported_model_commands)
                error = f"Unknown prompt command '/{key}'. Supported / commands: {', '.join('/' + one for one in supported)}."
        return ""

    return SLASH_BLOCK_RE.sub(replace, prompt), wgp_options, model_options, has_options, error


def _floor_overlap(frame_count: int, step: int, offset: int) -> int:
    frame_count = max(0, int(frame_count))
    if frame_count == 0:
        return 0
    step = max(1, int(step))
    offset = max(0, int(offset))
    if offset == 0:
        return frame_count // step * step
    return 0 if frame_count < offset else (frame_count - offset) // step * step + offset


def resolve_window_geometry(output_frames: int, overlap_frames: int, discard_last_frames: int, minimum: int, step: int, *, frame_offset: int = 1, overlap_offset: int = 1, max_overlap: int | None = None, available_overlap: int | None = None, preserve_exact_output_frames: bool = False) -> dict:
    output_frames = max(1, int(output_frames))
    overlap_frames = max(0, int(overlap_frames))
    discard_last_frames = max(0, int(discard_last_frames))
    if overlap_frames == 0:
        overlaps = [0]
    else:
        overlap_limit = overlap_frames if max_overlap is None else max(overlap_frames, int(max_overlap))
        if available_overlap is not None:
            overlap_limit = min(overlap_limit, max(0, int(available_overlap)))
        overlap_limit = _floor_overlap(overlap_limit, step, overlap_offset)
        preferred_overlap = _floor_overlap(min(overlap_frames, overlap_limit), step, overlap_offset)
        overlaps = list(range(preferred_overlap, overlap_limit + 1, max(1, int(step)))) if preferred_overlap > 0 else [0]

    candidates = []
    for overlap in overlaps:
        frame_num = normalize_frame_count(output_frames + overlap + discard_last_frames, minimum, step, frame_offset)
        trim_last_frames = frame_num - overlap - discard_last_frames - output_frames
        candidates.append((trim_last_frames, frame_num, overlap))
    trim_last_frames, frame_num, overlap_frames = min(candidates)
    if not preserve_exact_output_frames:
        output_frames += trim_last_frames
        trim_last_frames = 0
    return {
        "output_frames": output_frames,
        "overlap_frames": overlap_frames,
        "discard_last_frames": discard_last_frames,
        "trim_last_frames": trim_last_frames,
        "frame_num": frame_num,
    }


def _window(prompt: str, output_frames: int, overlap_frames: int, discard_last_frames: int, model_options: dict | None, minimum: int, step: int, *, frame_offset: int = 1, overlap_offset: int = 1, max_overlap: int | None = None, available_overlap: int | None = None, new_shot: bool = False, preserve_exact_output_frames: bool = False) -> dict:
    geometry = resolve_window_geometry(output_frames, 0 if new_shot else overlap_frames, discard_last_frames, minimum, step, frame_offset=frame_offset, overlap_offset=overlap_offset, max_overlap=max_overlap, available_overlap=available_overlap, preserve_exact_output_frames=preserve_exact_output_frames)
    return {
        "prompt": prompt,
        **geometry,
        "new_shot": bool(new_shot),
        "model_options": dict(model_options or {}),
    }


def build_extension_window(prompt: str, *, window_size: int, overlap_frames: int, discard_last_frames: int = 0, minimum: int, step: int, frame_offset: int = 1, preserve_exact_output_frames: bool = False) -> dict:
    overlap_offset = overlap_frames % max(1, step)
    return _window(prompt, max(1, window_size - overlap_frames - discard_last_frames), overlap_frames, discard_last_frames, {}, minimum, step, frame_offset=frame_offset, overlap_offset=overlap_offset, preserve_exact_output_frames=preserve_exact_output_frames)


def build_default_window_plan(*, total_frames: int, window_size: int, default_overlap: int, discard_last_frames: int, minimum: int, step: int, frame_offset: int = 1, overlap_offset: int = 1, max_overlap: int | None = None, first_window_overlap: int = 0, first_window_available_overlap: int | None = None, preserve_exact_output_frames: bool = False) -> list[dict]:
    total_frames = max(1, int(total_frames))
    window_size = normalize_frame_count(window_size, minimum, step, frame_offset)
    first_window_overlap = max(0, int(first_window_overlap))
    if first_window_available_overlap is not None:
        first_window_overlap = min(first_window_overlap, max(0, int(first_window_available_overlap)))
    first_window_overlap = _floor_overlap(first_window_overlap, step, overlap_offset)
    first_window_capacity = max(1, window_size - first_window_overlap)
    sliding = total_frames > first_window_capacity
    first_discard = max(0, int(discard_last_frames)) if sliding else 0
    first_output = min(total_frames, max(1, first_window_capacity - first_discard))
    windows = [_window("", first_output, first_window_overlap, first_discard, {}, minimum, step, frame_offset=frame_offset, overlap_offset=overlap_offset, max_overlap=max_overlap, available_overlap=first_window_available_overlap, preserve_exact_output_frames=preserve_exact_output_frames)]
    consumed = windows[0]["output_frames"]
    while consumed < total_frames:
        output_frames = min(total_frames - consumed, max(1, window_size - default_overlap - discard_last_frames))
        window = _window("", output_frames, default_overlap, discard_last_frames, {}, minimum, step, frame_offset=frame_offset, overlap_offset=overlap_offset, max_overlap=max_overlap, available_overlap=consumed, preserve_exact_output_frames=preserve_exact_output_frames)
        windows.append(window)
        consumed += window["output_frames"]
    return windows


def clone_loras_slists(slists):
    if slists is None:
        return None
    cloned = {}
    for key, value in slists.items():
        if isinstance(value, dict):
            cloned[key] = clone_loras_slists(value)
        elif isinstance(value, list):
            cloned[key] = value[:]
        else:
            cloned[key] = value
    return cloned


def prepare_loras_mult_windows(frame_scheduler: dict | None, activated_loras, num_inference_steps: int, guidance_phases: int, *, base_loras_slists=None, model_switch_phase: int = 1, store_slists: bool = False, lora_multiplier_branches=None) -> str | None:
    if frame_scheduler is None or not frame_scheduler.get("active", False):
        return None
    from shared.utils.loras_mutipliers import parse_loras_multipliers
    for idx, window in enumerate(frame_scheduler["windows"], start=1):
        window_loras_multipliers = window.get("loras_multipliers", "")
        if len(window_loras_multipliers) > 0:
            if len(activated_loras) == 0:
                return f"Sliding window {idx} uses /loras_mult but no LoRA is selected."
            _, window_loras_slists, errors = parse_loras_multipliers(window_loras_multipliers, len(activated_loras), num_inference_steps, nb_phases=guidance_phases, merge_slist=clone_loras_slists(base_loras_slists), model_switch_phase=model_switch_phase, lora_multiplier_branches=lora_multiplier_branches)
            if len(errors) > 0:
                return f"Error parsing /loras_mult for Sliding window {idx}: {errors}"
            if store_slists:
                window["loras_slists"] = window_loras_slists
    return None


def build_frame_scheduler(
    prompts: list[str],
    *,
    total_frames: int,
    fps: float,
    window_size: int,
    default_overlap: int,
    minimum: int,
    step: int,
    frame_offset: int = 1,
    overlap_offset: int = 1,
    max_overlap: int | None = None,
    supported_model_commands=(),
    allow_new_shot: bool = False,
    first_window_overlap_frames: int = 0,
    discard_last_frames: int = 0,
    preserve_exact_output_frames: bool = False,
) -> tuple[dict, str | None]:
    supported_model_commands = {str(command).strip().lower().lstrip("/") for command in supported_model_commands or [] if str(command).strip()}
    default_overlap, error = normalize_overlap(default_overlap, step, overlap_offset)
    if error is not None:
        return {}, error
    discard_last_frames = max(0, discard_last_frames)
    first_window_overlap_frames = max(0, first_window_overlap_frames)
    parsed_prompts, parsed = [], []
    any_options = False
    any_duration = False
    for prompt in prompts:
        stripped, wgp_options, model_options, has_options, error = _parse_options(prompt, supported_model_commands=supported_model_commands, allow_new_shot=allow_new_shot, fps=fps, total_frames=total_frames, step=step, overlap_offset=overlap_offset, default_overlap=default_overlap)
        if error is not None:
            return {}, error
        parsed_prompts.append(stripped.strip())
        parsed.append((stripped.strip(), wgp_options, model_options))
        any_options = any_options or has_options
        any_duration = any_duration or "duration_frames" in wgp_options

    if not any_options:
        return {"active": False, "prompts": parsed_prompts, "model_commands": sorted(supported_model_commands)}, None

    windows = []
    consumed = 0
    for idx, (prompt, wgp_options, model_options) in enumerate(parsed, start=1):
        overlap = wgp_options.get("overlap_frames", default_overlap)
        if idx == 1:
            overlap = min(overlap, first_window_overlap_frames)
        duration = wgp_options.get("duration_frames")
        if duration is None:
            remaining = total_frames - consumed
            if remaining <= 0:
                return {}, f"Sliding window {idx} would generate no frame because previous windows already consume the requested frame count. Unable to start generation: please specify shorter /duration values for the previous sliding windows or increase the total number of frames."
            duration = min(remaining, max(1, window_size - overlap - discard_last_frames))
        window = _window(prompt, duration, overlap, discard_last_frames, model_options, minimum, step, frame_offset=frame_offset, overlap_offset=overlap_offset, max_overlap=max_overlap, available_overlap=first_window_overlap_frames if idx == 1 else consumed, new_shot=bool(wgp_options.get("new_shot", False)), preserve_exact_output_frames=preserve_exact_output_frames)
        if "loras_multipliers" in wgp_options:
            window["loras_multipliers"] = wgp_options["loras_multipliers"]
        windows.append(window)
        consumed += window["output_frames"]

    while not any_duration and consumed < total_frames and windows:
        duration = min(total_frames - consumed, max(1, window_size - default_overlap - discard_last_frames))
        windows.append(_window(windows[-1]["prompt"], duration, default_overlap, discard_last_frames, {}, minimum, step, frame_offset=frame_offset, overlap_offset=overlap_offset, max_overlap=max_overlap, available_overlap=consumed, preserve_exact_output_frames=preserve_exact_output_frames))
        consumed += windows[-1]["output_frames"]

    return {
        "active": True,
        "prompts": [window["prompt"] for window in windows],
        "windows": windows,
        "predicted_total_frames": sum(window["output_frames"] for window in windows),
        "requested_total_frames": total_frames,
        "default_window_size": normalize_frame_count(window_size, minimum, step, frame_offset),
        "default_overlap_frames": default_overlap,
        "overlap_offset": overlap_offset,
        "minimum": minimum,
        "step": step,
        "frame_offset": frame_offset,
        "model_commands": sorted(supported_model_commands),
    }, None
