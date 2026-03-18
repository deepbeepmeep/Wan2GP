"""Module-oriented structure namespace with curated symbol exports."""

from __future__ import annotations

from importlib import import_module

from source.media.structure.frame_ops import create_neutral_frame

__all__ = [
    "api",
    "frame_ops",
    "create_neutral_frame",
    "create_structure_guidance_video",
    "download_and_extract_motion_frames",
    "load_structure_video_frames",
    "create_composite_guidance_video",
]

_MODULES = {
    "api": "source.media.structure.api",
    "frame_ops": "source.media.structure.frame_ops",
    "loading": "source.media.structure.loading",
    "preprocessors": "source.media.structure.preprocessors",
    "generation": "source.media.structure.generation",
    "segments": "source.media.structure.segments",
    "tracker": "source.media.structure.tracker",
    "download": "source.media.structure.download",
    "io": "source.media.structure.io",
}
_ATTR_EXPORTS = {
    "create_structure_guidance_video": ("source.media.structure.generation", "create_structure_guidance_video"),
    "download_and_extract_motion_frames": ("source.media.structure.download", "download_and_extract_motion_frames"),
    "load_structure_video_frames": ("source.media.structure.loading", "load_structure_video_frames"),
    "create_composite_guidance_video": ("source.media.structure.compositing", "create_composite_guidance_video"),
    "validate_structure_video_configs": ("source.media.structure.compositing", "validate_structure_video_configs"),
    "load_structure_video_frames_with_range": ("source.media.structure.frame_ops", "load_structure_video_frames_with_range"),
    "calculate_segment_stitched_position": ("source.media.structure.segments", "calculate_segment_stitched_position"),
    "extract_segment_structure_guidance": ("source.media.structure.segments", "extract_segment_structure_guidance"),
    "segment_has_structure_overlap": ("source.media.structure.segments", "segment_has_structure_overlap"),
}


def __getattr__(name: str):
    attr_export = _ATTR_EXPORTS.get(name)
    if attr_export:
        module = import_module(attr_export[0])
        value = getattr(module, attr_export[1])
        globals()[name] = value
        return value
    if name in _MODULES:
        module = import_module(_MODULES[name])
        globals()[name] = module
        return module
    if name == "create_neutral_frame":
        return create_neutral_frame
    raise AttributeError(name)


def __dir__():
    return sorted(set(__all__) | set(_MODULES) | set(_ATTR_EXPORTS))
