from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import gradio as gr

from . import constants


def choose_resolution(budget_label: str) -> str:
    resolutions = {"256p": "448x256", "320p": "576x320", "384p": "672x384", "480p": "832x480", "540p": "960x544", "720p": "1280x720", "900p": "1600x896", "1080p": "1920x1088"}
    try:
        return resolutions[str(budget_label)]
    except KeyError as exc:
        raise gr.Error(f"Unsupported Output Resolution: {budget_label}") from exc


def format_time_token(seconds: float | None) -> str:
    if seconds in (None, ""):
        return "end"
    total_centiseconds = max(0, int(round(float(seconds) * 100.0)))
    total_seconds, centiseconds = divmod(total_centiseconds, 100)
    minutes, seconds_only = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    seconds_text = f"{seconds_only:02d}" if centiseconds <= 0 else f"{seconds_only:02d}.{centiseconds:02d}"
    if hours > 0:
        return f"{hours:02d}h{minutes:02d}m{seconds_text}s"
    return f"{minutes:02d}m{seconds_text}s"


def get_process_filename_token(process_name: str) -> str:
    words = str(process_name or "").strip().split()
    if len(words) == 0:
        return "process"
    token = "".join(char for char in words[0].lower() if char.isalnum() or char in {"-", "_"})
    return token or "process"


def _supported_suffix(preferred_suffix: str, default_container: str) -> str:
    preferred_container = str(preferred_suffix or "").strip().lower().lstrip(".")
    if preferred_container in constants.SUPPORTED_OUTPUT_CONTAINERS:
        return f".{preferred_container}"
    fallback_container = str(default_container or "mp4").strip().lower() or "mp4"
    return f".{fallback_container}" if fallback_container in constants.SUPPORTED_OUTPUT_CONTAINERS else ".mp4"


def build_auto_output_path(source_path: str, process_name: str, ratio_text: str, output_resolution: str, start_seconds: float | None, end_seconds: float | None, output_dir: str | None = None, *, has_outpaint: bool = False, default_container: str = "mp4") -> str:
    source = Path(source_path)
    process_token = get_process_filename_token(process_name)
    resolution_suffix = str(output_resolution or "").strip() or "res"
    start_suffix = format_time_token(start_seconds)
    end_suffix = format_time_token(end_seconds)
    target_dir = source.parent if not output_dir else Path(output_dir)
    output_suffix = _supported_suffix(source.suffix, default_container)
    name_parts = [source.stem, process_token]
    if has_outpaint:
        name_parts.append(str(ratio_text or "").replace(":", "x") or "ratio")
    name_parts.extend([resolution_suffix, start_suffix, end_suffix])
    return str(target_dir / f"{'_'.join(name_parts)}{output_suffix}")


def make_output_variant(output: Path, *, notify: Callable[[str], None] | None = None) -> str:
    for index in range(2, 10000):
        candidate = output.with_name(f"{output.stem}_{index}{output.suffix}")
        if not candidate.exists():
            if notify is not None:
                notify(f"Output file already exists. Using {candidate}")
            return str(candidate)
    raise gr.Error(f"Unable to find a free output filename for {output}")


def list_continuation_output_paths(output_path: str) -> list[str]:
    output = Path(output_path)
    base_stem = f"{output.stem}_continue"
    candidates: list[tuple[int, str]] = []
    for child in output.parent.glob(f"{base_stem}*{output.suffix}"):
        if not child.is_file():
            continue
        if child.stem == base_stem:
            candidates.append((1, str(child)))
            continue
        prefix = base_stem + "_"
        if not child.stem.startswith(prefix):
            continue
        suffix = child.stem[len(prefix):]
        if suffix.isdigit():
            candidates.append((int(suffix), str(child)))
    return [path for _, path in sorted(candidates)]


def make_continuation_output_path(output_path: str) -> str:
    output = Path(output_path)
    existing_paths = list_continuation_output_paths(str(output))
    if len(existing_paths) == 0:
        return str(output.with_name(f"{output.stem}_continue{output.suffix}"))
    max_index = 1
    base_stem = f"{output.stem}_continue"
    for existing_path in existing_paths:
        existing_stem = Path(existing_path).stem
        if existing_stem == base_stem:
            max_index = max(max_index, 1)
            continue
        suffix = existing_stem[len(base_stem) + 1:]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    for index in range(max_index + 1, 10000):
        variant = output.with_name(f"{output.stem}_continue_{index}{output.suffix}")
        if not variant.exists():
            return str(variant)
    raise gr.Error(f"Unable to find a free continuation filename for {output}")


def build_requested_output_path(source_path: str, output_path: str, process_name: str, ratio_text: str, output_resolution: str, start_seconds: float | None, end_seconds: float | None, *, has_outpaint: bool = False, default_container: str = "mp4") -> Path:
    output_text = str(output_path or "").strip()
    if len(output_text) == 0:
        output = Path(build_auto_output_path(source_path, process_name, ratio_text, output_resolution, start_seconds, end_seconds, has_outpaint=has_outpaint, default_container=default_container))
    elif output_text.endswith(("\\", "/")) or Path(output_text).is_dir():
        output = Path(build_auto_output_path(source_path, process_name, ratio_text, output_resolution, start_seconds, end_seconds, output_dir=output_text, has_outpaint=has_outpaint, default_container=default_container))
    else:
        output = Path(output_text)
    if not output.suffix:
        output = output.with_suffix(_supported_suffix("", default_container))
    elif output.suffix.lstrip(".").lower() not in constants.SUPPORTED_OUTPUT_CONTAINERS:
        supported_text = ", ".join(f".{container}" for container in sorted(constants.SUPPORTED_OUTPUT_CONTAINERS))
        raise gr.Error(f"Output File must use one of these container extensions: {supported_text}.")
    return output


def resolve_output_path(source_path: str, output_path: str, process_name: str, ratio_text: str, output_resolution: str, start_seconds: float | None, end_seconds: float | None, continue_enabled: bool, *, has_outpaint: bool = False, default_container: str = "mp4", notify: Callable[[str], None] | None = None) -> tuple[str, bool]:
    output = build_requested_output_path(source_path, output_path, process_name, ratio_text, output_resolution, start_seconds, end_seconds, has_outpaint=has_outpaint, default_container=default_container)
    if continue_enabled:
        return str(output), output.exists()
    if output.exists():
        return make_output_variant(output, notify=notify), False
    return str(output), False
