import re

H3_VIDEO_EXCERPTS_SETTING = "h3_video_excerpt_positions"
H3_AUDIO_EXCERPTS_SETTING = "h3_audio_excerpt_positions"
H3_EXCERPT_DEFAULT_SECONDS = 3.0
H3_EXCERPT_MIN_SECONDS = 2.0
H3_EXCERPTS_MAX_SECONDS = 15.0
H3_EXCERPTS_MAX_COUNT = 3

_EXCERPT = re.compile(r"(\d+(?:\.\d+)?s|\d+)(?:/(\d+(?:\.\d+)?)s?)?", re.IGNORECASE)

H3_EXCERPT_SETTINGS = [
    {"id": H3_VIDEO_EXCERPTS_SETTING, "name": "Reference Video Excerpt Positions", "label": "Reference Positions from Control Video (frames or seconds, optional /duration, e.g. 3 5.2s/4s 12s)", "type": "text", "default": "", "video_prompt_type": "1"},
    {"id": H3_AUDIO_EXCERPTS_SETTING, "name": "Soundtrack Excerpt Positions", "label": "Audio Reference Positions from Reference-Video Soundtrack (frames or seconds, optional /duration, e.g. 3 5.2s/4s 12s)", "type": "text", "default": "", "audio_prompt_type": "1"},
]


def parse_excerpts(text, fps, media_seconds, label, video_frames=False):
    """Return up to three (start, duration) excerpts in seconds, centred on each position and kept inside the media.
    With video_frames, durations use the nearest H3 reference length of 17n+5 frames that fits in the media."""
    tokens = (text or "").replace(",", " ").split()
    if not tokens:
        raise ValueError(f"{label}: enter up to {H3_EXCERPTS_MAX_COUNT} positions, for example 3 5.2s/4s 12s")
    if len(tokens) > H3_EXCERPTS_MAX_COUNT:
        raise ValueError(f"{label}: at most {H3_EXCERPTS_MAX_COUNT} excerpts (found {len(tokens)})")
    excerpts = []
    for token in tokens:
        match = _EXCERPT.fullmatch(token)
        if match is None:
            raise ValueError(f"{label}: invalid position '{token}'; use a frame number or seconds such as 5s or 3.2s, optionally followed by a duration such as /2.5s")
        position, duration = match.groups()
        center = float(position[:-1]) if position[-1] in "sS" else (int(position) - 1) / fps
        duration = H3_EXCERPT_DEFAULT_SECONDS if duration is None else float(duration)
        if center > media_seconds:
            raise ValueError(f"{label}: position '{token}' is beyond the end of the media ({media_seconds:.2f}s)")
        if duration < H3_EXCERPT_MIN_SECONDS:
            raise ValueError(f"{label}: excerpt '{token}' must last at least {H3_EXCERPT_MIN_SECONDS:g}s")
        if duration > media_seconds:
            raise ValueError(f"{label}: excerpt '{token}' is longer than the media ({media_seconds:.2f}s)")
        if video_frames:
            duration = (min(round((duration * fps - 5) / 17), int((media_seconds * fps - 5) // 17)) * 17 + 5) / fps
        excerpts.append((min(max(center - duration / 2, 0.0), media_seconds - duration), duration))
    total = sum(duration for _, duration in excerpts)
    if total > H3_EXCERPTS_MAX_SECONDS:
        raise ValueError(f"{label}: excerpts total {total:.2f}s, above the {H3_EXCERPTS_MAX_SECONDS:g}s limit")
    return excerpts
