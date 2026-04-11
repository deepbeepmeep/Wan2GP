"""Friendly labels and compact display helpers for log output."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

TASK_TYPE_LABELS: dict[str, str] = {
    "annotated_image_edit": "Annotated Image Edit",
    "comfy": "ComfyUI",
    "edit_video_orchestrator": "Edit Video",
    "extract_frame": "Extract Frame",
    "flux": "Flux",
    "generate_video": "Generate Video",
    "hunyuan": "Hunyuan",
    "i2v": "Image to Video",
    "i2v_22": "Image to Video 2.2",
    "image_inpaint": "Image Inpaint",
    "individual_travel_segment": "Travel Segment",
    "inpaint_frames": "Inpaint Frames",
    "join_clips_final_stitch": "Join Clips Final Stitch",
    "join_clips_orchestrator": "Join Clips",
    "join_clips_segment": "Join Clips Segment",
    "join_final_stitch": "Join Clips Final Stitch",
    "ltx2": "LTX Video 2",
    "ltxv": "LTX Video",
    "magic_edit": "Magic Edit",
    "qwen_image": "Qwen Image",
    "qwen_image_2512": "Qwen Image 2512",
    "qwen_image_edit": "Qwen Image Edit",
    "qwen_image_hires": "Qwen Image Hi-Res",
    "qwen_image_style": "Qwen Image Style",
    "rife_interpolate": "RIFE Interpolate",
    "rife_interpolate_images": "RIFE Interpolate",
    "t2v": "Text to Video",
    "t2v_22": "Text to Video 2.2",
    "travel_orchestrator": "Travel",
    "travel_segment": "Travel Segment",
    "travel_stitch": "Travel Stitch",
    "vace": "VACE",
    "vace_21": "VACE 2.1",
    "vace_22": "VACE 2.2",
    "wan_2_2_t2i": "Text to Image 2.2",
    "z_image_turbo": "Z Image Turbo",
    "z_image_turbo_i2i": "Z Image Turbo I2I",
}

_TASK_TYPE_SHORT_NAMES: dict[str, str] = {
    "annotated_image_edit": "annotated",
    "comfy": "comfy",
    "edit_video_orchestrator": "edit",
    "extract_frame": "extract",
    "flux": "flux",
    "generate_video": "generate",
    "hunyuan": "hunyuan",
    "i2v": "i2v",
    "i2v_22": "i2v",
    "image_inpaint": "inpaint",
    "individual_travel_segment": "travel",
    "inpaint_frames": "inpaint",
    "join_clips_final_stitch": "join",
    "join_clips_orchestrator": "join",
    "join_clips_segment": "join",
    "join_final_stitch": "join",
    "ltx2": "ltx",
    "ltxv": "ltx",
    "magic_edit": "magic",
    "qwen_image": "qwen",
    "qwen_image_2512": "qwen",
    "qwen_image_edit": "qwen",
    "qwen_image_hires": "qwen",
    "qwen_image_style": "qwen",
    "rife_interpolate": "rife",
    "rife_interpolate_images": "rife",
    "t2v": "t2v",
    "t2v_22": "t2v",
    "travel_orchestrator": "travel",
    "travel_segment": "travel",
    "travel_stitch": "travel",
    "vace": "vace",
    "vace_21": "vace",
    "vace_22": "vace",
    "wan_2_2_t2i": "t2i",
    "z_image_turbo": "zimg",
    "z_image_turbo_i2i": "zimg",
}

__all__ = [
    "TASK_TYPE_LABELS",
    "friendly_child_id",
    "friendly_task_id",
    "model_label",
    "rel_path",
    "task_type_label",
]


def task_type_label(task_type: str) -> str:
    """Return the friendly label for a task type."""
    if not task_type:
        return ""
    return TASK_TYPE_LABELS.get(task_type, task_type.replace("_", " ").title())


def friendly_task_id(task_id: str, task_type: str) -> str:
    """Return a compact task identifier for human-facing log lines."""
    short_type = _TASK_TYPE_SHORT_NAMES.get(task_type, _fallback_short_type(task_type))
    raw_task_id = str(task_id or "").strip()
    if not raw_task_id:
        return short_type
    compact_id = raw_task_id.replace("-", "")[:8]
    return f"{short_type}#{compact_id}" if compact_id else short_type


def friendly_child_id(parent_task_id: str, parent_task_type: str, kind: str, idx: int | None = None) -> str:
    """Return a friendly hierarchical log-only child identifier."""
    base = friendly_task_id(parent_task_id, parent_task_type)
    suffix = f"{kind}{idx:02d}" if idx is not None else str(kind)
    return f"{base}/{suffix}" if suffix else base


def model_label(internal_name: str) -> str:
    """Return a user-facing model name for an internal model key."""
    if not internal_name:
        return internal_name or ""

    try:
        from source.runtime.wgp_bridge import get_model_name

        display_name = get_model_name(internal_name)
        if not display_name.startswith("Unknown model"):
            return display_name
    except (ImportError, RuntimeError, AttributeError):
        pass

    return internal_name.replace("_", " ").title()


def rel_path(path: str | Path) -> str:
    """Return a repo-relative path when possible, otherwise the basename."""
    if not path:
        return ""

    original = Path(path)
    repo_root = _repo_root()
    candidate = original if original.is_absolute() else (Path.cwd() / original)

    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        resolved = candidate

    if repo_root is not None:
        try:
            return resolved.relative_to(repo_root).as_posix()
        except ValueError:
            pass

    return original.name or resolved.name or str(original)


def _fallback_short_type(task_type: str) -> str:
    cleaned = (task_type or "task").replace("_", "-").strip("-")
    return cleaned[:24] or "task"


@lru_cache(maxsize=1)
def _repo_root() -> Path | None:
    current = Path(__file__).resolve()
    for directory in (current.parent, *current.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    return None
