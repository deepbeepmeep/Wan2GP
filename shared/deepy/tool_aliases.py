# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Former argument names of toolbox actions, still accepted: each maps to its current name (seconds take an _s suffix)."""

RENAMED_ARGUMENTS = {
    "extract_image": {"time_seconds": "time_s"},
    "extract_video": {"start_time": "start_s", "end_time": "end_s", "duration": "duration_s"},
    "extract_audio": {"start_time": "start_s", "end_time": "end_s", "duration": "duration_s"},
    "inspect_audio": {"start_time": "start_s", "duration": "duration_s", "min_silence": "min_silence_s", "preview_seconds": "preview_s"},
    "inspect_video": {"start_time_seconds": "start_s", "end_time_seconds": "end_s", "min_frames_between_samples": "min_gap_frames", "min_seconds_between_samples": "min_gap_s", "prompt": "question"},
    "inspect_media": {"time_seconds": "time_s", "prompt": "question"},
    "merge_videos": {"video_ids": "media_ids"}, "remux_media": {"video": "video_id"},
}
RENAMED_ITEM_ARGUMENTS = {("inspect_media", "media_inputs"): {"time_seconds": "time_s"}}


def _renamed(values, renames, where):
    result = {}
    for name, value in values.items():
        current = renames.get(name, name)
        if current != name and current in values:
            raise ValueError(f"{where}{name} is the former name of {current}: pass only {current}.")
        result[current] = value
    return result


def canonical_arguments(action, arguments):
    """The arguments of a toolbox action with former names replaced by the current ones."""
    if not isinstance(arguments, dict):
        return arguments
    result = _renamed(arguments, RENAMED_ARGUMENTS.get(action, {}), "")
    for (item_action, key), renames in RENAMED_ITEM_ARGUMENTS.items():
        if item_action == action and isinstance(result.get(key), list):
            result[key] = [_renamed(item, renames, f"{key}[{index}].") if isinstance(item, dict) else item for index, item in enumerate(result[key])]
    return result
