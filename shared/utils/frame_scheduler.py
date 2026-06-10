from __future__ import annotations

import math
import re


WGP_SLASH_COMMANDS = {"duration", "overlap", "new_shot", "loras_mult"}
SLASH_BLOCK_RE = re.compile(r"\[\s*/\s*([^\]]+?)\s*\]", re.IGNORECASE)


def has_slash_commands(prompts: list[str]) -> bool:
    return any(SLASH_BLOCK_RE.search(prompt or "") is not None for prompt in prompts)


def normalize_frame_count(frame_count: int, minimum: int, step: int) -> int:
    frame_count = max(minimum, frame_count)
    step = max(1, step)
    return math.ceil(max(0, frame_count - 1) / step) * step + 1 if step > 1 else frame_count


def normalize_output_frame_count(frame_count: int, minimum: int, step: int) -> int:
    frame_count = max(minimum, frame_count)
    step = max(1, step)
    if step <= 1:
        return frame_count
    lower = max(minimum, ((frame_count - 1) // step) * step + 1)
    upper = normalize_frame_count(frame_count, minimum, step)
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


def _window(prompt: str, output_frames: int, overlap_frames: int, discard_last_frames: int, model_options: dict | None, minimum: int, step: int, *, new_shot: bool = False) -> dict:
    overlap_frames = max(0, overlap_frames)
    output_frames = normalize_output_frame_count(output_frames, minimum, step)
    discard_last_frames = max(0, discard_last_frames)
    frame_num = normalize_frame_count(output_frames + overlap_frames + discard_last_frames, minimum, step)
    return {
        "prompt": prompt,
        "output_frames": output_frames,
        "overlap_frames": overlap_frames,
        "discard_last_frames": frame_num - output_frames - overlap_frames,
        "frame_num": frame_num,
        "new_shot": bool(new_shot),
        "model_options": dict(model_options or {}),
    }


def build_extension_window(prompt: str, *, window_size: int, overlap_frames: int, discard_last_frames: int = 0, minimum: int, step: int) -> dict:
    return _window(prompt, max(1, window_size - overlap_frames - discard_last_frames), overlap_frames, discard_last_frames, {}, minimum, step)


def clone_loras_slists(slists):
    return {key: value[:] if isinstance(value, list) else value for key, value in slists.items()} if slists is not None else None


def prepare_loras_mult_windows(frame_scheduler: dict | None, activated_loras, num_inference_steps: int, guidance_phases: int, *, base_loras_slists=None, model_switch_phase: int = 1, store_slists: bool = False) -> str | None:
    if frame_scheduler is None or not frame_scheduler.get("active", False):
        return None
    from shared.utils.loras_mutipliers import parse_loras_multipliers
    for idx, window in enumerate(frame_scheduler["windows"], start=1):
        window_loras_multipliers = window.get("loras_multipliers", "")
        if len(window_loras_multipliers) > 0:
            if len(activated_loras) == 0:
                return f"Sliding window {idx} uses /loras_mult but no LoRA is selected."
            _, window_loras_slists, errors = parse_loras_multipliers(window_loras_multipliers, len(activated_loras), num_inference_steps, nb_phases=guidance_phases, merge_slist=clone_loras_slists(base_loras_slists), model_switch_phase=model_switch_phase)
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
    overlap_offset: int = 1,
    supported_model_commands=(),
    allow_new_shot: bool = False,
    first_window_overlap_frames: int = 0,
    discard_last_frames: int = 0,
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
        window = _window(prompt, duration, overlap, discard_last_frames, model_options, minimum, step, new_shot=bool(wgp_options.get("new_shot", False)))
        if "loras_multipliers" in wgp_options:
            window["loras_multipliers"] = wgp_options["loras_multipliers"]
        windows.append(window)
        consumed += window["output_frames"]

    while not any_duration and consumed < total_frames and windows:
        duration = min(total_frames - consumed, max(1, window_size - default_overlap - discard_last_frames))
        windows.append(_window(windows[-1]["prompt"], duration, default_overlap, discard_last_frames, {}, minimum, step))
        consumed += windows[-1]["output_frames"]

    return {
        "active": True,
        "prompts": [window["prompt"] for window in windows],
        "windows": windows,
        "predicted_total_frames": sum(window["output_frames"] for window in windows),
        "requested_total_frames": total_frames,
        "default_window_size": normalize_frame_count(window_size, minimum, step),
        "default_overlap_frames": default_overlap,
        "overlap_offset": overlap_offset,
        "minimum": minimum,
        "step": step,
        "model_commands": sorted(supported_model_commands),
    }, None
